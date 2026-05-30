#!/usr/bin/env python3
"""Zapisz best_model.joblib z istniejacych modeli (bez pelnego treningu)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
from best_model import save_best_model
from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader import load_hr
from evaluate import evaluate_models
from features import get_X_y
from sklearn.model_selection import train_test_split


def main() -> int:
    meta_path = MODELS_DIR / "metadata.joblib"
    if not meta_path.is_file():
        print("Brak models/metadata.joblib — uruchom najpierw trening.")
        return 1

    meta = joblib.load(meta_path)
    pipelines = {}
    SKIP = {"metadata", "best_model", "preprocessing_pipeline"}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if path.stem in SKIP:
            continue
        pipelines[path.stem] = joblib.load(path)

    if not pipelines:
        print("Brak zapisanych modeli w models/.")
        return 1

    df = load_hr()
    X, y, _, _ = get_X_y(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    metrics = evaluate_models(pipelines, X_test, y_test)
    save_best_model(
        pipelines,
        metrics,
        threshold_sources=meta.get("threshold_sources"),
        oof_f1_scores=meta.get("tuning_f1_at_threshold", meta.get("oof_f1_scores")),
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
