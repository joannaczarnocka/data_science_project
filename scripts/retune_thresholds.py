"""Ponowny dobór progów na zapisanych modelach (bez pelnego treningu)."""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", module="tensorflow")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
from config import F1_TARGET, MODELS_DIR
from data_loader import load_hr
from features import get_X_y
from sklearn.model_selection import train_test_split
from threshold import ThresholdClassifier, select_threshold

from config import RANDOM_STATE, TEST_SIZE


def main() -> None:
    df = load_hr()
    X, y, _, _ = get_X_y(df)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    meta_path = MODELS_DIR / "metadata.joblib"
    skip = {"metadata", "ensemble"}
    updated = 0

    for path in sorted(MODELS_DIR.glob("*.joblib")):
        name = path.stem
        if name in skip:
            continue
        try:
            wrapped = joblib.load(path)
        except Exception as exc:
            print(f"  pomijam {name}: {exc}")
            continue
        if not isinstance(wrapped, ThresholdClassifier):
            print(f"  pomijam {name}: nie ThresholdClassifier")
            continue

        pipe = wrapped.pipeline
        thr, f1, src = select_threshold(pipe, X_train, y_train, F1_TARGET)
        new_wrapped = ThresholdClassifier(pipe, threshold=thr, threshold_source=src)
        joblib.dump(new_wrapped, path)
        mark = "OK" if f1 >= F1_TARGET else "ponizej celu"
        print(f"  {name}: F1={f1:.3f}, prog={thr:.3f}, zrodlo={src} [{mark}]")
        updated += 1

    if meta_path.exists():
        meta = joblib.load(meta_path)
        meta["threshold_tuning"] = f"F1>={F1_TARGET}: oof -> train -> smote_train (retune)"
        joblib.dump(meta, meta_path)

    print(f"\nZaktualizowano progi: {updated} modeli.")


if __name__ == "__main__":
    main()
