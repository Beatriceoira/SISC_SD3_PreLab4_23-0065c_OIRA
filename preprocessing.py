"""Landmark extraction, clip I/O, manifest handling and signer-disjoint splitting.

Pipeline (identical for live webcam, video files and recorded clips):
    BGR frame -> horizontal flip (selfie view, as in the starter) -> MediaPipe Hands
    -> two fixed hand slots: slot 0 = Left, slot 1 = Right (handedness preserved)
    -> raw landmarks (2, 21, 3) + presence mask (2,)   [missing hand = zeros + mask False]

Raw coordinates are stored as (x * aspect, y, z * aspect) with aspect = width / height,
so x and y share one unit (image height). Normalization happens in features.py.

CLI:
    python preprocessing.py record --lang ASL --label HELLO --signer S01 --consent yes
    python preprocessing.py split  --manifest ../03_Dataset/dataset_manifest.csv
    python preprocessing.py build  --manifest ../03_Dataset/dataset_manifest.csv --lang ASL
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "03_Dataset"
MODELS_DIR = ROOT / "04_Models"
RESULTS_DIR = ROOT / "06_Results"

NUM_LM = 21
SEQ_LEN = 30                 # frames per resampled clip
MIN_HAND_FRAMES = 5          # clips with fewer hand frames are rejected
SLOT_NAMES = ("Left", "Right")
UNKNOWN = "UNKNOWN"

MANIFEST_COLUMNS = ["path", "language", "label", "signer_id", "source", "consent",
                    "split", "dynamic", "hands", "notes"]


# --------------------------------------------------------------------------- landmarks
def assign_slots(results):
    """Map MediaPipe results to fixed slots. Returns [(slot, hand_landmarks), ...].

    slot 0 = Left, slot 1 = Right (taken from MediaPipe handedness on the mirrored frame).
    If both detections claim the same side, the lower-confidence one takes the free slot.
    """
    lms = getattr(results, "multi_hand_landmarks", None) or []
    hd = getattr(results, "multi_handedness", None) or []
    pairs = []
    for i, lm in enumerate(lms):
        if i < len(hd):
            cls = hd[i].classification[0]
            label, score = cls.label, cls.score
        else:
            label, score = "Right", 0.0
        pairs.append((score, SLOT_NAMES.index(label), lm))
    taken, out = set(), []
    for score, slot, lm in sorted(pairs, key=lambda p: -p[0]):
        if slot in taken:
            slot = 1 - slot
            if slot in taken:
                continue
        taken.add(slot)
        out.append((slot, lm))
    return out


def landmarks_to_arrays(results, aspect):
    """-> raw (2, 21, 3) float32, mask (2,) bool."""
    raw = np.zeros((2, NUM_LM, 3), np.float32)
    mask = np.zeros(2, bool)
    for slot, lm in assign_slots(results):
        raw[slot] = np.array([[p.x * aspect, p.y, p.z * aspect] for p in lm.landmark], np.float32)
        mask[slot] = True
    return raw, mask


def make_hands(static=False, complexity=1, det=0.6, track=0.6):
    import mediapipe as mp
    return mp.solutions.hands.Hands(static_image_mode=static, max_num_hands=2,
                                    model_complexity=complexity,
                                    min_detection_confidence=det,
                                    min_tracking_confidence=track)


# --------------------------------------------------------------------------- clips
def extract_video(path):
    """Run MediaPipe over a video file. -> raw (T,2,21,3), mask (T,2), fps."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raws, masks = [], []
    hands = make_hands()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            r, m = landmarks_to_arrays(res, w / h)
            raws.append(r)
            masks.append(m)
    finally:
        cap.release()
        hands.close()
    if not raws:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(raws), np.stack(masks), float(fps)


def load_clip(path, cache_dir=None):
    """Load a clip from .npz (recorded landmarks) or a video file (cached extraction)."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        d = np.load(path, allow_pickle=False)
        return d["raw"], d["mask"].astype(bool), float(d["fps"])
    cache_file = None
    if cache_dir is not None:
        key = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
        cache_file = Path(cache_dir) / f"{path.stem}_{key}.npz"
        if cache_file.exists():
            d = np.load(cache_file, allow_pickle=False)
            return d["raw"], d["mask"].astype(bool), float(d["fps"])
    raw, mask, fps = extract_video(path)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, raw=raw, mask=mask, fps=fps)
    return raw, mask, fps


def trim_and_resample(raw, mask, fps, seq_len=SEQ_LEN):
    """Trim leading/trailing no-hand frames, resample to seq_len frames.

    Returns raw (seq_len,2,21,3), mask (seq_len,2), duration_seconds  (or None if too short).
    """
    idx = np.where(mask.any(axis=1))[0]
    if len(idx) < MIN_HAND_FRAMES:
        return None
    a, b = idx[0], idx[-1]
    raw, mask = raw[a:b + 1], mask[a:b + 1]
    n = len(raw)
    sel = np.linspace(0, n - 1, seq_len).round().astype(int)
    return raw[sel], mask[sel], n / fps


# --------------------------------------------------------------------------- splits
def assign_splits(df, seed=42, ratios=(0.6, 0.2, 0.2)):
    """Fill blank `split` cells so that no signer appears in more than one split."""
    df = df.copy()
    blank = df["split"].isna() | (df["split"].astype(str).str.strip() == "")
    if not blank.any():
        return df
    signers = sorted(df.loc[blank, "signer_id"].astype(str).unique())
    if len(signers) < 3:
        raise ValueError(f"Need >= 3 distinct signers for a signer-disjoint split, got {len(signers)}: {signers}")
    rng = np.random.RandomState(seed)
    rng.shuffle(signers)
    n = len(signers)
    n_val = max(1, int(round(n * ratios[1])))
    n_test = max(1, int(round(n * ratios[2])))
    n_train = n - n_val - n_test
    while n_train < 1:
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
        n_train = n - n_val - n_test
    where = {}
    for i, s in enumerate(signers):
        where[s] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    df.loc[blank, "split"] = df.loc[blank, "signer_id"].astype(str).map(where)
    return df


def check_signer_disjoint(df):
    """Raise if a signer appears in more than one split. Returns {split: set(signers)}."""
    by = {s: set(g["signer_id"].astype(str)) for s, g in df.groupby("split")}
    names = sorted(by)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = by[a] & by[b]
            if both:
                raise ValueError(f"Signer leakage between {a} and {b}: {sorted(both)}")
    return by


def report_splits(df, lang):
    print(f"\n== {lang} split report ==")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    for s, sig in check_signer_disjoint(df).items():
        print(f"  {s}: signers={sorted(sig)}")
    labels = set(df["label"]) - {UNKNOWN}
    for s in ("train", "val", "test"):
        miss = labels - set(df[df.split == s]["label"])
        if miss:
            print(f"  WARNING: split '{s}' has no clips for {sorted(miss)}")


# --------------------------------------------------------------------------- dataset build
def read_manifest(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    miss = [c for c in MANIFEST_COLUMNS[:9] if c not in df.columns]
    if miss:
        raise ValueError(f"Manifest missing columns: {miss}")
    return df


def build_dataset(manifest_path, lang, out_path=None, label_map_path=None):
    manifest_path = Path(manifest_path)
    lang = lang.upper()
    full = read_manifest(manifest_path)
    full["language"] = full["language"].str.upper()
    sub = full[full["language"] == lang].copy()
    if sub.empty:
        raise ValueError(f"No {lang} rows in manifest")
    bad = sub[sub["consent"].str.lower() != "yes"]
    if len(bad):
        print(f"WARNING: skipping {len(bad)} rows without consent=yes")
        sub = sub[sub["consent"].str.lower() == "yes"]
    sub = assign_splits(sub)
    # write assigned splits back so the split is reproducible
    full.loc[sub.index, "split"] = sub["split"]
    full.to_csv(manifest_path, index=False)
    report_splits(sub, lang)

    lm_path = Path(label_map_path) if label_map_path else DATASET_DIR / "label_map.json"
    if lm_path.exists():
        known = {c["label"] for c in json.loads(lm_path.read_text()).get(lang, {}).get("classes", [])}
        unk = set(sub["label"]) - known - {UNKNOWN}
        if unk:
            print(f"WARNING: labels not in label_map.json[{lang}]: {sorted(unk)}")

    cache_dir = DATASET_DIR / "processed" / f"cache_{lang.lower()}"
    raws, masks, durs, kept = [], [], [], []
    for i, (_, row) in enumerate(sub.iterrows()):
        p = Path(row["path"])
        p = p if p.is_absolute() else manifest_path.parent / p
        try:
            raw, mask, fps = load_clip(p, cache_dir)
            out = trim_and_resample(raw, mask, fps)
        except Exception as e:  # keep going, report at the end
            print(f"  skip {row['path']}: {e}")
            continue
        if out is None:
            print(f"  skip {row['path']}: fewer than {MIN_HAND_FRAMES} hand frames")
            continue
        r, m, d = out
        raws.append(r); masks.append(m); durs.append(d); kept.append(row)
        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(sub)}")
    if not raws:
        raise RuntimeError("No usable clips")
    k = pd.DataFrame(kept)
    out_path = Path(out_path) if out_path else DATASET_DIR / "processed" / f"{lang.lower()}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, raw=np.stack(raws), mask=np.stack(masks), durations=np.array(durs, np.float32),
        labels=k["label"].to_numpy(str), signers=k["signer_id"].to_numpy(str),
        splits=k["split"].to_numpy(str), dynamic=k["dynamic"].astype(int).to_numpy(),
        hands=k["hands"].astype(int).to_numpy(), paths=k["path"].to_numpy(str))
    print(f"Saved {len(raws)} clips -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- recording
def record_cli(args):
    """Record landmark-only clips from the webcam (no raw video is stored -> privacy friendly)."""
    import cv2
    if args.consent.lower() != "yes":
        raise SystemExit("Refusing to record: pass --consent yes only after the signer gave written consent.")
    lang = args.lang.upper()
    out_dir = Path(args.out) / lang / args.signer
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")
    hands = make_hands()
    recording, buf_r, buf_m, t_start, count = False, [], [], 0.0, 0
    print("SPACE = start/stop clip, Q = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            r, m = landmarks_to_arrays(res, w / h)
            if recording:
                buf_r.append(r); buf_m.append(m)
            for slot, lm in assign_slots(res):
                import mediapipe as mp
                mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
            status = "REC" if recording else "ready"
            cv2.putText(frame, f"{lang} {args.label} signer={args.signer} [{status}] clips={count}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("Record landmarks", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == 32:
                if not recording:
                    recording, buf_r, buf_m, t_start = True, [], [], time.perf_counter()
                else:
                    recording = False
                    dur = time.perf_counter() - t_start
                    if len(buf_r) < MIN_HAND_FRAMES:
                        print("clip too short, discarded")
                        continue
                    fps = len(buf_r) / max(dur, 1e-3)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    fn = out_dir / f"{args.label}_{stamp}.npz"
                    np.savez_compressed(fn, raw=np.stack(buf_r), mask=np.stack(buf_m), fps=fps)
                    row = {"path": str(fn.relative_to(manifest.parent)) if manifest.parent in fn.parents else str(fn),
                           "language": lang, "label": args.label, "signer_id": args.signer,
                           "source": "self-recorded landmarks", "consent": "yes", "split": "",
                           "dynamic": args.dynamic, "hands": args.hands, "notes": ""}
                    pd.DataFrame([row], columns=MANIFEST_COLUMNS).to_csv(
                        manifest, mode="a", header=not manifest.exists(), index=False)
                    count += 1
                    print(f"saved {fn.name} ({len(buf_r)} frames, {fps:.1f} fps)")
    finally:
        cap.release(); hands.close(); cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("--manifest", default=str(DATASET_DIR / "dataset_manifest.csv"))
    b = sub.add_parser("build"); b.add_argument("--manifest", default=str(DATASET_DIR / "dataset_manifest.csv"))
    b.add_argument("--lang", required=True); b.add_argument("--out")
    r = sub.add_parser("record")
    r.add_argument("--lang", required=True); r.add_argument("--label", required=True)
    r.add_argument("--signer", required=True); r.add_argument("--consent", default="no")
    r.add_argument("--dynamic", type=int, default=0); r.add_argument("--hands", type=int, default=1)
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--manifest", default=str(DATASET_DIR / "dataset_manifest.csv"))
    r.add_argument("--out", default=str(DATASET_DIR / "sample_data"))
    a = ap.parse_args()
    if a.cmd == "split":
        df = read_manifest(a.manifest)
        for lang in sorted(df["language"].str.upper().unique()):
            m = df["language"].str.upper() == lang
            df.loc[m] = assign_splits(df[m])
            report_splits(df[m], lang)
        df.to_csv(a.manifest, index=False)
    elif a.cmd == "build":
        build_dataset(a.manifest, a.lang, a.out)
    else:
        record_cli(a)


if __name__ == "__main__":
    main()
