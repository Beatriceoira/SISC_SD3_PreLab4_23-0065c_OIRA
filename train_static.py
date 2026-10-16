"""Train the STATIC (single-frame) classifier for one language: RandomForest vs MLP.

Trains only on clips flagged dynamic=0. Model selection uses validation macro-F1 (frame level);
the chosen model is evaluated once on val and test (frame level and clip level).
Clips labelled UNKNOWN are never trained on; test-split UNKNOWN clips measure rejection.

    python train_static.py --lang ASL
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import (confusion_matrix, macro_prf, metric_rows, save_confusion, unknown_row,
                        update_config, write_metrics)
from features import FRAME_DIM, frame_features
from preprocessing import DATASET_DIR, MODELS_DIR, RESULTS_DIR, UNKNOWN


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def frame_samples(D, idx, cls_index):
    X, y, g = [], [], []
    for i in idx:
        feats, _, _ = frame_features(D["raw"][i], D["mask"][i])
        keep = D["mask"][i].any(axis=1)
        X.append(feats[keep])
        y += [cls_index[D["labels"][i]]] * int(keep.sum())
        g += [i] * int(keep.sum())
    return np.concatenate(X), np.array(y), np.array(g)


def clip_probs(model, X, g, n_cls):
    """Mean class probability over each clip's frames -> (clips, probs)."""
    P = np.zeros((X.shape[0], n_cls))
    P[:, model.classes_] = model.predict_proba(X)
    ids = np.unique(g)
    return ids, np.stack([P[g == i].mean(0) for i in ids])


def candidates(seed):
    return {
        "rf_100": RandomForestClassifier(100, random_state=seed, n_jobs=-1),
        "rf_300": RandomForestClassifier(300, random_state=seed, n_jobs=-1),
        "mlp_128_64": make_pipeline(StandardScaler(), MLPClassifier(
            (128, 64), max_iter=500, early_stopping=True, random_state=seed)),
        "mlp_256_128": make_pipeline(StandardScaler(), MLPClassifier(
            (256, 128), max_iter=500, early_stopping=True, random_state=seed)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--data")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--reject-thr", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    lang = a.lang.upper()
    D = load_npz(a.data or DATASET_DIR / "processed" / f"{lang.lower()}.npz")

    known = D["labels"] != UNKNOWN
    static = known & (D["dynamic"] == 0)
    classes = sorted(str(c) for c in set(D["labels"][static & (D["splits"] == "train")]))
    if len(classes) < 2:
        raise SystemExit(f"Need >= 2 static classes in the train split, found {classes}")
    cls_index = {c: i for i, c in enumerate(classes)}
    idx = {s: np.where(static & (D["splits"] == s) & np.isin(D["labels"], classes))[0]
           for s in ("train", "val", "test")}
    print(f"{lang} static classes ({len(classes)}): {classes}")
    print({s: len(v) for s, v in idx.items()}, "clips")
    Xtr, ytr, _ = frame_samples(D, idx["train"], cls_index)
    Xva, yva, gva = frame_samples(D, idx["val"], cls_index)

    # ---- model selection on validation macro-F1
    sel, best, best_f1 = [], None, -1
    for name, clf in candidates(a.seed).items():
        clf.fit(Xtr, ytr)
        cm = confusion_matrix(yva, clf.predict(Xva), len(classes))
        m = macro_prf(cm)
        sel.append(dict(model=name, val_accuracy=m["accuracy"], val_macro_f1=m["macro_f1"]))
        print(f"  {name:12s} val acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")
        if m["macro_f1"] > best_f1:
            best, best_name, best_f1 = clf, name, m["macro_f1"]
    res = Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sel).to_csv(res / f"model_selection_{lang}_static.csv", index=False)
    print(f"selected: {best_name}")

    # ---- final evaluation
    rows = []
    n = len(classes)
    for split in ("val", "test"):
        if len(idx[split]) == 0:
            print(f"  (no {split} clips)"); continue
        X, y, g = frame_samples(D, idx[split], cls_index)
        cm = confusion_matrix(y, best.predict(X), n)
        rows += metric_rows("static", "frame", split, cm, classes)
        ids, P = clip_probs(best, X, g, n)
        truth = np.array([cls_index[D["labels"][i]] for i in ids])
        cm_clip = confusion_matrix(truth, P.argmax(1), n)
        rows += metric_rows("static", "clip", split, cm_clip, classes)
        save_confusion(cm_clip, classes, res / "confusion_matrices" / f"{lang}_static_{split}.png",
                       f"{lang} static ({best_name}) - {split} (clip level)")
        print(f"  {split}: frame macroF1={macro_prf(cm)['macro_f1']:.3f} "
              f"clip macroF1={macro_prf(cm_clip)['macro_f1']:.3f} acc={macro_prf(cm_clip)['accuracy']:.3f}")
        if split == "test":
            unk = np.where((D["labels"] == UNKNOWN) & (D["splits"] == "test"))[0]
            pu = []
            if len(unk):
                Xu, _, gu = frame_samples(D, unk, {UNKNOWN: 0})
                pu = clip_probs(best, Xu, gu, n)[1].max(1)
            rows.append(unknown_row("static", "test", P.max(1), pu, a.reject_thr))
    write_metrics(res / f"metrics_{lang}.csv", rows, "static")

    md = Path(a.models_dir)
    md.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=best, classes=classes, lang=lang, feature_dim=FRAME_DIM,
                     reject_thr=a.reject_thr, name=best_name), md / f"{lang.lower()}_static.joblib")
    update_config(md, lang, "static", dict(selected=best_name, classes=classes, seed=a.seed,
                                           candidates=sel, reject_thr=a.reject_thr,
                                           feature_dim=FRAME_DIM, train_clips=int(len(idx["train"]))))
    print(f"saved {md / (lang.lower() + '_static.joblib')}")


if __name__ == "__main__":
    main()
