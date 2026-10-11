import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from analysis.audio_io import load_wav
from analysis.deep_model import MODEL_PATH, CRNN, compute_features, segment
from analysis.robust import ML_SR, augment, normalize_for_ml


def load_files(root):
    files, labels = [], []
    for label, sub in ((0, "real"), (1, "fake")):
        found = sorted(glob.glob(os.path.join(root, sub, "**", "*.wav"), recursive=True))
        print(f"  {sub}: {len(found)} файла")
        files += found
        labels += [label] * len(found)
    return files, np.array(labels)


def build_segments(files, labels, kind, n_aug=0, seed=0):
    rng = np.random.default_rng(seed)
    X, y, owner = [], [], []
    for i, (path, lab) in enumerate(zip(files, labels)):
        try:
            a = load_wav(path, max_seconds=30)
        except Exception as exc:
            print(f"    пропуснат {path}: {exc}")
            continue
        clean = normalize_for_ml(a.samples, a.sr)
        for x in [clean] + [augment(clean, rng) for _ in range(n_aug)]:
            segs, _ = segment(compute_features(x, ML_SR, kind))
            X.append(segs)
            y += [lab] * len(segs)
            owner += [i] * len(segs)
        if (i + 1) % 100 == 0:
            print(f"    обработени {i + 1}/{len(files)}")
    return np.concatenate(X), np.array(y, dtype=np.float32), np.array(owner)


def evaluate(model, X, y, batch=64):
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            probs.append(torch.sigmoid(model(torch.from_numpy(X[s:s + batch]))).numpy())
    return np.concatenate(probs)


def save_curves(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(hist["loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    a1.plot(ep, hist["loss"], color="#2a78d6", label="обучение")
    a1.plot(ep, hist["val_loss"], color="#eb6834", label="валидация")
    a1.set_title("Загуба (BCE)"); a1.set_xlabel("Епоха"); a1.legend()
    a2.plot(ep, hist["acc"], color="#2a78d6", label="обучение")
    a2.plot(ep, hist["val_acc"], color="#eb6834", label="валидация")
    a2.set_title("Точност"); a2.set_xlabel("Епоха"); a2.set_ylim(0, 1.02); a2.legend()
    for a in (a1, a2):
        a.grid(alpha=.3); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--features", choices=["mfcc", "logmel"], default="mfcc")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", type=int, default=1, help="брой аугментирани копия на файл (0 = без)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Зареждане на файловете…")
    files, labels = load_files(args.data)
    if len(set(labels)) < 2 or min(np.bincount(labels)) < 4:
        print("Нужни са поне 4 файла във всяка от папките dataset/real и dataset/fake.")
        return 1
    idx = np.arange(len(files))
    tr_f, va_f = train_test_split(idx, test_size=0.2, stratify=labels, random_state=args.seed)
    print("Извличане на признаци…")
    Xtr, ytr, _ = build_segments([files[i] for i in tr_f], labels[tr_f], args.features,
                                 n_aug=args.aug, seed=args.seed)
    Xva, yva, own_va = build_segments([files[i] for i in va_f], labels[va_f], args.features,
                                      n_aug=min(args.aug, 1), seed=args.seed + 1)
    print(f"  прозорци: обучение {len(Xtr)}, валидация {len(Xva)}; размер на вход {Xtr.shape[1:]}")

    model = CRNN(n_feat=Xtr.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модел CRNN (CNN + BiLSTM): {n_params:,} параметъра")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_weight = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    hist = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    best_auc, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot, correct = 0.0, 0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            xb = torch.from_numpy(Xtr[b])
            xb = xb + 0.05 * torch.randn_like(xb)
            if np.random.rand() < 0.5:
                f0 = np.random.randint(0, xb.shape[1] - 8)
                xb[:, f0:f0 + np.random.randint(1, 8)] = 0
            yb = torch.from_numpy(ytr[b])
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
            correct += ((logits > 0).float() == yb).sum().item()
        sched.step()
        p_va = evaluate(model, Xva, yva)
        val_loss = float(nn.functional.binary_cross_entropy(torch.tensor(p_va).clamp(1e-6, 1 - 1e-6),
                                                            torch.tensor(yva)))
        val_acc = float(((p_va > 0.5) == yva).mean())
        auc = roc_auc_score(yva, p_va) if len(set(yva)) > 1 else 0.5
        hist["loss"].append(tot / len(Xtr)); hist["acc"].append(correct / len(Xtr))
        hist["val_loss"].append(val_loss); hist["val_acc"].append(val_acc)
        print(f"Епоха {epoch:3d}/{args.epochs}  загуба {tot / len(Xtr):.4f}  точност {correct / len(Xtr):.3f}  "
              f"вал. загуба {val_loss:.4f}  вал. точност {val_acc:.3f}  AUC {auc:.3f}  ({time.time() - t0:.1f} s)")
        if auc > best_auc or (auc == best_auc and val_acc >= max(hist["val_acc"])):
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    p_va = evaluate(model, Xva, yva)
    file_p = np.array([p_va[own_va == o].mean() for o in np.unique(own_va)])
    file_y = np.array([yva[own_va == o][0] for o in np.unique(own_va)])
    print("\nВалидация по ФАЙЛОВЕ (най-добрата епоха):")
    print(classification_report(file_y, file_p > 0.5, target_names=["естествен", "синтетичен"], digits=3,
                                zero_division=0))
    print("Матрица на объркване:\n", confusion_matrix(file_y, file_p > 0.5))
    if len(set(file_y)) > 1:
        print(f"ROC AUC (файлове): {roc_auc_score(file_y, file_p):.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    meta = {"features": args.features, "n_feat": int(Xtr.shape[1]), "epochs": args.epochs,
            "input": "ml16k", "aug": args.aug,
            "val_auc": float(best_auc), "n_files": len(files), "params": n_params,
            "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    torch.save({"state_dict": model.state_dict(), "meta": meta}, MODEL_PATH)
    curves = os.path.join(os.path.dirname(MODEL_PATH), "training_curves.png")
    save_curves(hist, curves)
    print(f"\nМоделът е записан в {MODEL_PATH}\nГрафиката на обучението: {curves}")
    return 0


if __name__ == "__main__":
    sys.exit(main())