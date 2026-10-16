"""Real-time inference: hand tracking, static/temporal prediction, stabilizer FSM, TTS, latency.

Decision flow per frame:
    HandTracker -> (raw, mask) -> SignRecognizer.update()
        motion energy < threshold  -> static classifier (mean prob of last STATIC_K frames)
        motion energy >= threshold -> temporal classifier (last WINDOW_SEC, resampled to 30 frames)
        confidence < reject threshold -> UNKNOWN
    SignStabilizer (FSM) -> display label + at most one emitted transcript token per sign.

Stabilizer FSM:
    NO_HANDS --hands appear--> SEARCHING
    SEARCHING --same confident label N frames--> CANDIDATE -> COMMITTED (emit once)
    COMMITTED --different label stable N frames--> COMMITTED (emit new label)
    COMMITTED --hands absent RELEASE frames, or UNKNOWN N frames--> NO_HANDS / SEARCHING (re-armed)
    A label is never re-emitted while COMMITTED, and not within COOLDOWN seconds of the last emit.
"""
import csv
import platform
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation import read_config
from features import motion_energy, frame_features, sequence_features
from preprocessing import (MODELS_DIR, RESULTS_DIR, SEQ_LEN, UNKNOWN, landmarks_to_arrays,
                           make_hands)

WINDOW_SEC = 1.5
STATIC_K = 8
MIN_WINDOW_FRAMES = 10


@dataclass
class Prediction:
    label: object          # None = no hands, "UNKNOWN", or a class name
    conf: float = 0.0
    source: str = "none"   # static | temporal | none
    motion: float = 0.0


# --------------------------------------------------------------------------- tracking
class HandTracker:
    def __init__(self, complexity=1):
        self.hands = make_hands(complexity=complexity)

    def process(self, frame_bgr):
        """-> (mirrored frame, mediapipe results, raw (2,21,3), mask (2,))."""
        import cv2
        frame = cv2.flip(frame_bgr, 1)
        h, w = frame.shape[:2]
        res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        raw, mask = landmarks_to_arrays(res, w / h)
        return frame, res, raw, mask

    def close(self):
        self.hands.close()


# --------------------------------------------------------------------------- recognizer
class SignRecognizer:
    def __init__(self, lang, models_dir=MODELS_DIR):
        self.lang = lang.upper()
        self.static = self.temporal = None
        self.messages = []
        md = Path(models_dir)
        cfg = read_config(md, self.lang)
        self.motion_thr = (cfg.get("temporal") or {}).get("motion_threshold")
        self.reject_thr = 0.6
        sp = md / f"{self.lang.lower()}_static.joblib"
        if sp.exists():
            import joblib
            self.static = joblib.load(sp)
            self.reject_thr = self.static.get("reject_thr", self.reject_thr)
        tp = md / f"{self.lang.lower()}_model.pt"
        if tp.exists():
            try:
                import torch
                from train_temporal import SignRNN
                ck = torch.load(tp, map_location="cpu", weights_only=False)
                m = SignRNN(ck["in_dim"], len(ck["classes"]), ck["hidden"], ck["layers"], ck["cell"])
                m.load_state_dict(ck["state_dict"]); m.eval()
                self.temporal = dict(model=m, torch=torch, **{k: ck[k] for k in ("classes", "mean", "std")})
                self.reject_thr = ck.get("reject_thr", self.reject_thr)
            except Exception as e:
                self.messages.append(f"temporal model not loaded: {e}")
        if self.static is None and self.temporal is None:
            self.messages.append(f"No {self.lang} models in {md} - train first")
        self.buf = deque()   # (t, raw, mask)

    @property
    def ready(self):
        return self.static is not None or self.temporal is not None

    def reset(self):
        self.buf.clear()

    # -- helpers
    def _window(self):
        """Trim no-hand ends of the buffer; return (raw, mask, duration) resampled to SEQ_LEN."""
        t = np.array([b[0] for b in self.buf])
        raw = np.stack([b[1] for b in self.buf]); mask = np.stack([b[2] for b in self.buf])
        hit = np.where(mask.any(1))[0]
        if len(hit) < MIN_WINDOW_FRAMES:
            return None
        a, b = hit[0], hit[-1]
        t, raw, mask = t[a:b + 1], raw[a:b + 1], mask[a:b + 1]
        dur = max(t[-1] - t[0], 1e-3)
        targets = np.linspace(t[0], t[-1], SEQ_LEN)
        sel = np.abs(t[None, :] - targets[:, None]).argmin(1)
        return raw[sel], mask[sel], dur

    def _static_probs(self):
        recent = list(self.buf)[-STATIC_K:]
        raw = np.stack([r[1] for r in recent]); mask = np.stack([r[2] for r in recent])
        keep = mask.any(1)
        if not keep.any():
            return None
        feats, _, _ = frame_features(raw[keep], mask[keep])
        return self.static["model"].predict_proba(feats).mean(0), self.static["model"].classes_

    def _temporal_probs(self, raw, mask, dur):
        T = self.temporal
        x = sequence_features(raw, mask, dur / (SEQ_LEN - 1))
        x = ((x - T["mean"]) / T["std"]).astype(np.float32)[None]
        with T["torch"].no_grad():
            return T["torch"].softmax(T["model"](T["torch"].from_numpy(x)), dim=1)[0].numpy()

    # -- main entry
    def update(self, raw, mask, t=None):
        t = time.perf_counter() if t is None else t
        self.buf.append((t, raw.copy(), mask.copy()))
        while self.buf and t - self.buf[0][0] > WINDOW_SEC:
            self.buf.popleft()
        if not mask.any() and not any(b[2].any() for b in list(self.buf)[-3:]):
            return Prediction(None)
        if not self.ready:
            return Prediction(UNKNOWN)
        win = self._window()
        if win is None:
            return Prediction(UNKNOWN)
        raw_w, mask_w, dur = win
        motion = motion_energy(raw_w, mask_w, dur / (SEQ_LEN - 1))
        use_temporal = self.temporal is not None and (
            self.static is None or (self.motion_thr is not None and motion >= self.motion_thr))
        if use_temporal:
            p = self._temporal_probs(raw_w, mask_w, dur)
            label, conf, src = self.temporal["classes"][int(p.argmax())], float(p.max()), "temporal"
        else:
            r = self._static_probs()
            if r is None:
                return Prediction(UNKNOWN, motion=motion)
            p, cls = r
            label, conf, src = self.static["classes"][int(cls[int(p.argmax())])], float(p.max()), "static"
        if conf < self.reject_thr:
            label = UNKNOWN
        return Prediction(label, conf, src, motion)


# --------------------------------------------------------------------------- FSM
class SignStabilizer:
    def __init__(self, conf_thr=0.6, confirm_frames=8, release_frames=10, cooldown_s=1.0):
        self.conf_thr, self.confirm, self.release, self.cooldown = conf_thr, confirm_frames, release_frames, cooldown_s
        self.reset()

    def reset(self):
        self.state, self.cand, self.cand_n = "NO_HANDS", None, 0
        self.committed, self.no_hand, self.unk_n, self.last_emit = None, 0, 0, -1e9

    def step(self, pred, now=None):
        """-> dict(state, label, conf, emit)  (emit is a label once per sign, else None)."""
        now = time.perf_counter() if now is None else now
        if pred.label is None:
            self.no_hand += 1
            if self.no_hand >= self.release:
                self.state, self.cand, self.cand_n, self.committed = "NO_HANDS", None, 0, None
            return dict(state=self.state, label="-", conf=0.0, emit=None)
        self.no_hand = 0
        if self.state == "NO_HANDS":
            self.state = "SEARCHING"
        label = pred.label if (pred.label != UNKNOWN and pred.conf >= self.conf_thr) else UNKNOWN
        if label == UNKNOWN:
            self.unk_n += 1
            self.cand, self.cand_n = None, 0
            if self.unk_n >= self.confirm:
                self.committed, self.state = None, "SEARCHING"
            elif self.state == "CANDIDATE":
                self.state = "SEARCHING"
            return dict(state=self.state, label=UNKNOWN, conf=pred.conf, emit=None)
        self.unk_n = 0
        self.cand_n = self.cand_n + 1 if label == self.cand else 1
        self.cand = label
        if label == self.committed:
            self.state = "COMMITTED"
            return dict(state=self.state, label=label, conf=pred.conf, emit=None)
        self.state = "CANDIDATE"
        emit = None
        if self.cand_n >= self.confirm and now - self.last_emit >= self.cooldown:
            emit, self.committed, self.last_emit, self.state = label, label, now, "COMMITTED"
        return dict(state=self.state, label=label, conf=pred.conf, emit=emit)


# --------------------------------------------------------------------------- latency / FPS
class LatencyMeter:
    def __init__(self):
        self.lat, self.t_first, self.t_last = [], None, None

    def record(self, latency_s, now=None):
        now = time.perf_counter() if now is None else now
        self.t_first = self.t_first if self.t_first is not None else now
        self.t_last = now
        self.lat.append(latency_s)

    def summary(self):
        n = len(self.lat)
        span = (self.t_last - self.t_first) if n > 1 else 0.0
        return dict(frames=n, avg_latency_ms=1000 * float(np.mean(self.lat)) if n else 0.0,
                    p95_latency_ms=1000 * float(np.percentile(self.lat, 95)) if n else 0.0,
                    measured_fps=(n - 1) / span if span > 0 else 0.0)

    def write_csv(self, lang, path=None):
        path = Path(path) if path else RESULTS_DIR / "latency_fps.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(language=lang, **self.summary(), machine=platform.processor() or platform.machine(),
                   os=platform.platform(), python=platform.python_version(),
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        return row


# --------------------------------------------------------------------------- speech
class Speaker:
    """Optional text-to-speech (pyttsx3). Picks an English voice for ASL and a Filipino/Tagalog
    voice for FSL if installed; otherwise falls back to the default voice and reports it."""

    def __init__(self):
        self.q = queue.Queue()
        self.ok = False
        self.voices = []
        try:
            import pyttsx3
            self._pyttsx3 = pyttsx3
            eng = pyttsx3.init()
            self.voices = [(v.id, (v.name or "") + " " + str(getattr(v, "languages", ""))) for v in eng.getProperty("voices")]
            eng.stop()
            self.ok = True
            threading.Thread(target=self._worker, daemon=True).start()
        except Exception as e:
            self.error = str(e)

    def voice_for(self, lang):
        keys = ("en",) if lang == "ASL" else ("fil", "tl", "phil", "tagalog")
        for vid, desc in self.voices:
            if any(k in desc.lower() or k in vid.lower() for k in keys):
                return vid, True
        return (self.voices[0][0] if self.voices else None), False

    def say(self, text, lang):
        if self.ok and text.strip():
            self.q.put((text, lang))

    def _worker(self):
        eng = self._pyttsx3.init()
        while True:
            text, lang = self.q.get()
            vid, exact = self.voice_for(lang)
            try:
                if vid:
                    eng.setProperty("voice", vid)
                eng.say(text); eng.runAndWait()
            except Exception:
                pass
