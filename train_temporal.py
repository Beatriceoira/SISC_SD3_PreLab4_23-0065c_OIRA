"""Train the TEMPORAL classifier (GRU vs LSTM) for one language on 30-frame clip sequences.

Input per clip: (30, 205) sequence features (features.sequence_features), standardized with
train-split mean/std. Trains on all non-UNKNOWN clips (static + dynamic). Model selection by
validation macro-F1. Also stores the static/dynamic motion threshold used at inference to
decide which classifier answers.

    python train_temporal.py --lang ASL            (add --quick for a fast smoke run)
Requires PyTorch (pip install torch).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluation import (confusion_matrix, macro_prf, metric_rows, save_confusion, unknown_row,
                        update_config, write_metrics)
from features import SEQ_DIM, motion_energy, sequence_features
from preprocessing import DATASET_DIR, MODELS_DIR, RESULTS_DIR, SEQ_LEN, UNKNOWN


class SignRNN(nn.Module):
    """GRU/LSTM encoder; head sees [last output, mean over time]."""

    def __init__(self, in_dim, n_classes, hidden=128, layers=2, cell="gru", dropout=0.3):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(torch.cat([out[:, -1], out.mean(1)], dim=1))


def build_sequences(D, idx):
    X = []
    for i in idx:
        dt = max(float(D["durations"][i]), 1e-3) / (SEQ_LEN - 1)
        X.append(sequence_features(D["raw"][i], D["mask"][i], dt))
    return np.stack(X) if X else np.zeros((0, SEQ_LEN, SEQ_DIM), np.float32)


@torch.no_grad()
def predict_probs(model, X, device):
    model.eval()
    out = []
    for s in range(0, len(X), 128):
        xb = torch.from_numpy(X[s:s + 128]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def train_one(cell, hidden, Xtr, ytr, Xva, yva, n_cls, epochs, patience, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SignRNN(Xtr.shape[2], n_cls, hidden, 2, cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    best = (-1.0, 1e9, None, 0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 32):
            b = perm[s:s + 32]
            xb = Xt[b].to(device)
            xb = xb + 0.05 * torch.randn_like(xb)            # noise augmentation
            loss = lossf(model(xb), yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        P = predict_probs(model, Xva, device)
        f1 = macro_prf(confusion_matrix(yva, P.argmax(1), n_cls))["macro_f1"]
        vloss = float(-np.log(P[np.arange(len(yva)), yva] + 1e-9).mean())
        if f1 > best[0] or (f1 == best[0] and vloss < best[1]):
            best = (f1, vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        elif ep - best[3] >= patience:
            break
    model.load_state_dict(best[2])
    return model, best[0], best[3] + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--data")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--reject-thr", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    lang = a.lang.upper()
    if a.quick:
        a.epochs, a.patience = 8, 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data or DATASET_DIR / "processed" / f"{lang.lower()}.npz", allow_pickle=False)
    D = {k: d[k] for k in d.files}

    known = D["labels"] != UNKNOWN
    classes = sorted(str(c) for c in set(D["labels"][known & (D["splits"] == "train")]))
    if len(classes) < 2:
        raise SystemExit("Need >= 2 classes in the train split")
    ci = {c: i for i, c in enumerate(classes)}
    idx = {s: np.where(known & (D["splits"] == s) & np.isin(D["labels"], classes))[0]
           for s in ("train", "val", "test")}
    print(f"{lang} temporal classes ({len(classes)}): {classes}  device={device}")
    X = {s: build_sequences(D, v) for s, v in idx.items()}
    y = {s: np.array([ci[D["labels"][i]] for i in v], np.int64) for s, v in idx.items()}
    if len(X["val"]) == 0:
        raise SystemExit("Validation split is empty (needed for model selection)")
    mean = X["train"].reshape(-1, SEQ_DIM).mean(0)
    std = X["train"].reshape(-1, SEQ_DIM).std(0) + 1e-6
    for s in X:
        X[s] = ((X[s] - mean) / std).astype(np.float32)

    sel, best, n = [], None, len(classes)
    for cell in ("gru", "lstm"):
        for hidden in (64, 128):
            name = f"{cell}_{hidden}"
            model, f1, ep = train_one(cell, hidden, X["train"], y["train"], X["val"], y["val"],
                                      n, a.epochs, a.patience, a.seed, device)
            sel.append(dict(model=name, val_macro_f1=f1, epochs=ep))
            print(f"  {name:9s} val macroF1={f1:.3f} (best epoch {ep})")
            if best is None or f1 > best[0]:
                best = (f1, name, cell, hidden, model)
    _, best_name, cell, hidden, model = best
    res = Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sel).to_csv(res / f"model_selection_{lang}_temporal.csv", index=False)
    print(f"selected: {best_name}")

    rows = []
    for split in ("val", "test"):
        if len(X[split]) == 0:
            continue
        P = predict_probs(model, X[split], device)
        cm = confusion_matrix(y[split], P.argmax(1), n)
        rows += metric_rows("temporal", "clip", split, cm, classes)
        save_confusion(cm, classes, res / "confusion_matrices" / f"{lang}_temporal_{split}.png",
                       f"{lang} temporal ({best_name}) - {split}")
        m = macro_prf(cm)
        print(f"  {split}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")
        if split == "test":
            unk = np.where((D["labels"] == UNKNOWN) & (D["splits"] == "test"))[0]
            pu = []
            if len(unk):
                Xu = ((build_sequences(D, unk) - mean) / std).astype(np.float32)
                pu = predict_probs(model, Xu, device).max(1)
            rows.append(unknown_row("temporal", "test", P.max(1), pu, a.reject_thr))
    write_metrics(res / f"metrics_{lang}.csv", rows, "temporal")

    # static-vs-dynamic motion threshold from TRAIN clips only
    tr = idx["train"]
    en = np.array([motion_energy(D["raw"][i], D["mask"][i], max(float(D["durations"][i]), 1e-3) / (SEQ_LEN - 1))
                   for i in tr])
    dyn = D["dynamic"][tr] == 1
    thr = float((np.median(en[~dyn]) + np.median(en[dyn])) / 2) if dyn.any() and (~dyn).any() else None

    md = Path(a.models_dir)
    md.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), cell=cell, hidden=hidden, layers=2, classes=classes,
                    mean=mean, std=std, in_dim=SEQ_DIM, seq_len=SEQ_LEN, lang=lang,
                    reject_thr=a.reject_thr), md / f"{lang.lower()}_model.pt")
    update_config(md, lang, "temporal", dict(selected=best_name, classes=classes, seed=a.seed,
                                             candidates=sel, epochs=a.epochs, patience=a.patience,
                                             seq_len=SEQ_LEN, feature_dim=SEQ_DIM,
                                             reject_thr=a.reject_thr, motion_threshold=thr,
                                             train_clips=int(len(tr))))
    print(f"saved {md / (lang.lower() + '_model.pt')}  motion_threshold={thr}")


if __name__ == "__main__":
    main()
