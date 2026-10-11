import argparse
import glob
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.ai_agent import MODEL_PATH, extract_ml_features
from analysis.audio_io import load_wav
from analysis.robust import augment, normalize_for_ml


def load_dataset(root, n_aug=1, seed=42):
    rng = np.random.default_rng(seed)
    X, y, groups, names = [], [], [], None
    file_id = 0
    for label, sub in ((0, "real"), (1, "fake")):
        files = sorted(glob.glob(os.path.join(root, sub, "**", "*.wav"), recursive=True))
        print(f"  {sub}: {len(files)} файла")
        for i, path in enumerate(files, 1):
            try:
                audio = load_wav(path, max_seconds=30)
                clean = normalize_for_ml(audio.samples, audio.sr)
                versions = [clean] + [augment(clean, rng) for _ in range(n_aug)]
                rows = [extract_ml_features(audio, samples=v) for v in versions]
            except Exception as exc:
                print(f"    пропуснат {path}: {exc}")
                continue
            if names is None:
                names = sorted(rows[0])
            for feats in rows:
                X.append([feats[n] for n in names])
                y.append(label)
                groups.append(file_id)
            file_id += 1
            if i % 50 == 0:
                print(f"    обработени {i}/{len(files)}")
    return np.array(X, dtype=float), np.array(y), np.array(groups), names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", choices=["rf", "logreg"], default="rf")
    ap.add_argument("--aug", type=int, default=1, help="брой аугментирани копия на файл (0 = без)")
    args = ap.parse_args()

    print("Извличане на признаци…")
    X, y, groups, names = load_dataset(args.data, n_aug=args.aug)
    if len(set(y)) < 2 or min(np.bincount(y)) < 5:
        print("Нужни са поне 5 файла във всяка от папките dataset/real и dataset/fake.")
        return 1
    X = np.nan_to_num(X)

    if args.model == "rf":
        model = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42))
        name = "Random Forest"
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        name = "Логистична регресия"

    folds = max(2, min(5, min(np.bincount(y)) // (args.aug + 1)))
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"\n{folds}-кратна кръстосана проверка ({name}):")
    print(classification_report(y, pred, target_names=["естествен", "синтетичен"], digits=3))
    print("Матрица на объркване:\n", confusion_matrix(y, pred))
    print(f"ROC AUC: {roc_auc_score(y, proba):.3f}")

    model.fit(X, y)
    if args.model == "rf":
        imp = model[-1].feature_importances_
        print("\nНай-важни признаци:")
        for i in np.argsort(imp)[::-1][:10]:
            print(f"  {names[i]:25s} {imp[i]:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "features": names, "name": name, "input": "ml16k", "aug": args.aug,
                 "n_real": len(set(groups[y == 0])), "n_fake": len(set(groups[y == 1])),
                 "n_rows": len(y)}, MODEL_PATH)
    print(f"\nМоделът е записан в {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())