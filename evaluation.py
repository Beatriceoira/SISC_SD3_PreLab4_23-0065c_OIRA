"""Metrics (hand-written so the formulas can be explained), confusion matrices, CSV/JSON helpers.

Macro F1 = mean over classes of 2PR/(P+R), with
    P = TP/(TP+FP)  (0 if nothing was predicted as that class)
    R = TP/(TP+FN)  (0 if the class has no test samples)
    F1 = 0 when P+R = 0
Undefined-class handling: a class with zero support AND zero predictions is excluded from the
macro mean (reported as `undefined_excluded`). A class with support but no predictions, or
predictions but no support, is kept with the zeros above so errors are never hidden.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_COLUMNS = ["scope", "level", "split", "row_type", "name", "precision", "recall", "f1",
                  "support", "accuracy", "note"]


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def macro_prf(cm):
    """-> dict(per_class=[(P,R,F1,support)], macro_p, macro_r, macro_f1, accuracy, defined, excluded)."""
    n = cm.shape[0]
    tp = np.diag(cm).astype(float)
    pred = cm.sum(0).astype(float)
    sup = cm.sum(1).astype(float)
    per, keep = [], []
    for c in range(n):
        p = tp[c] / pred[c] if pred[c] > 0 else 0.0
        r = tp[c] / sup[c] if sup[c] > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per.append((p, r, f, int(sup[c])))
        keep.append(sup[c] > 0 or pred[c] > 0)
    keep = np.array(keep)
    arr = np.array([x[:3] for x in per])
    macro = arr[keep].mean(0) if keep.any() else np.zeros(3)
    total = cm.sum()
    return dict(per_class=per, macro_p=macro[0], macro_r=macro[1], macro_f1=macro[2],
                accuracy=float(tp.sum() / total) if total else 0.0,
                defined=int(keep.sum()), excluded=int((~keep).sum()))


def metric_rows(scope, level, split, cm, classes):
    m = macro_prf(cm)
    rows = [dict(scope=scope, level=level, split=split, row_type="summary", name="macro",
                 precision=m["macro_p"], recall=m["macro_r"], f1=m["macro_f1"],
                 support=int(cm.sum()), accuracy=m["accuracy"],
                 note=f"classes_defined={m['defined']};undefined_excluded={m['excluded']}")]
    for c, (p, r, f, s) in zip(classes, m["per_class"]):
        rows.append(dict(scope=scope, level=level, split=split, row_type="class", name=c,
                         precision=p, recall=r, f1=f, support=s, accuracy="", note=""))
    return rows


def unknown_row(scope, split, max_known, max_unknown, thr):
    """Open-set check: known clips should be accepted (conf >= thr), UNKNOWN clips rejected."""
    acc = float((np.asarray(max_known) >= thr).mean()) if len(max_known) else float("nan")
    rej = float((np.asarray(max_unknown) < thr).mean()) if len(max_unknown) else float("nan")
    return dict(scope=scope, level="clip", split=split, row_type="unknown", name="unknown_rejection",
                precision="", recall=rej, f1="", support=len(max_unknown), accuracy=acc,
                note=f"thr={thr};recall=unknown_reject_rate;accuracy=known_accept_rate")


def write_metrics(path, rows, scope):
    """Replace all rows of `scope` in the CSV, keep other scopes (static/temporal share a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["scope"] != scope]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def save_confusion(cm, classes, png_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(png_path.with_suffix(".csv"))
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * n + 2), max(4, 0.7 * n + 1.5)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(png_path, dpi=150); plt.close(fig)


def update_config(models_dir, lang, section, values):
    """Merge values into 04_Models/training_config.json under [lang][section]."""
    path = Path(models_dir) / "training_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(path.read_text()) if path.exists() else {}
    cfg.setdefault(lang, {}).setdefault(section, {}).update(values)
    path.write_text(json.dumps(cfg, indent=2, default=str))


def read_config(models_dir, lang):
    path = Path(models_dir) / "training_config.json"
    return json.loads(path.read_text()).get(lang, {}) if path.exists() else {}
