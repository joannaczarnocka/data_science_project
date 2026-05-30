"""Wybór i zapis najlepszego modelu (kryterium: F1 na zbiorze testowym)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from config import MODELS_DIR, REPORTS_DIR

BEST_MODEL_FILE = "best_model.joblib"
BEST_MODEL_REPORT = "best_model.txt"

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "baseline": "Baseline (najczęstsza klasa)",
    "logistic_regression": "Logistic Regression",
    "balanced_rf": "Balanced Random Forest",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "stacking": "Stacking (BRF+XGB+CB+SVM / LR)",
    "svm": "SVM",
    "neural_network": "Neural Network (MLP)",
}


def display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())


def select_best_model_key(metrics: pd.DataFrame, metric: str = "f1") -> str:
    """Klucz modelu z najlepsza metryka na holdout (domyslnie F1)."""
    if metric not in metrics.columns:
        raise KeyError(f"Brak kolumny {metric!r} w metrics")
    return str(metrics[metric].idxmax())


def _build_best_model_info(
    model_key: str,
    metrics: pd.DataFrame,
    threshold_sources: dict | None = None,
    oof_f1_scores: dict | None = None,
    pipe=None,
) -> dict:
    row = metrics.loc[model_key]
    info = {
        "model_key": model_key,
        "display_name": display_name(model_key),
        "selection_metric": "f1",
        "selection_set": "test_holdout_20pct",
        "test_f1": float(row["f1"]),
        "test_precision": float(row["precision"]),
        "test_recall": float(row["recall"]),
        "test_roc_auc": float(row.get("roc_auc", float("nan"))),
        "test_accuracy": float(row.get("accuracy", float("nan"))),
        "source_file": f"{model_key}.joblib",
        "saved_as": BEST_MODEL_FILE,
    }
    if threshold_sources and model_key in threshold_sources:
        info["threshold_source"] = threshold_sources[model_key]
    if oof_f1_scores and model_key in oof_f1_scores:
        info["f1_at_threshold_tuning"] = float(oof_f1_scores[model_key])
    if pipe is not None and hasattr(pipe, "threshold"):
        info["decision_threshold"] = float(pipe.threshold)
    return info


def format_best_model_report(info: dict) -> str:
    """Tekstowy opis najlepszego modelu (terminal / reports/best_model.txt)."""
    lines = [
        "NAJLEPSZY MODEL (best_model)",
        "=" * 50,
        f"Model:              {info['display_name']} ({info['model_key']})",
        f"Kryterium wyboru:   najwyzszy F1 na zbiorze testowym (holdout 20%)",
        f"Plik pipeline:      models/{info['saved_as']}",
        f"Kopia z:            models/{info['source_file']}",
        "",
        "Metryki na TEST:",
        f"  F1:         {info['test_f1']:.4f}",
        f"  Precision:  {info['test_precision']:.4f}",
        f"  Recall:     {info['test_recall']:.4f}",
        f"  ROC-AUC:    {info['test_roc_auc']:.4f}",
        f"  Accuracy:   {info['test_accuracy']:.4f}",
    ]
    if "decision_threshold" in info:
        lines.append(f"  Prog dec.:  {info['decision_threshold']:.4f}")
    if "threshold_source" in info:
        lines.append(f"  Zrodlo progu: {info['threshold_source']}")
    if "f1_at_threshold_tuning" in info:
        lines.append(f"  F1@prog (tuning): {info['f1_at_threshold_tuning']:.4f}")
    lines.extend(
        [
            "",
            "Streamlit uzywa models/best_model.joblib jako model glowny.",
            "=" * 50,
        ]
    )
    return "\n".join(lines)


def print_best_model_summary(info: dict) -> None:
    print()
    print(format_best_model_report(info))
    print()


def save_best_model(
    pipelines: dict,
    metrics: pd.DataFrame,
    *,
    threshold_sources: dict | None = None,
    oof_f1_scores: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Zapisz najlepszy pipeline jako models/best_model.joblib,
    raport reports/best_model.txt i wpis w metadata.joblib.
    """
    model_key = select_best_model_key(metrics)
    if model_key not in pipelines:
        raise KeyError(f"Brak pipeline dla {model_key!r}")

    pipe = pipelines[model_key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best_path = MODELS_DIR / BEST_MODEL_FILE
    joblib.dump(pipe, best_path)

    info = _build_best_model_info(
        model_key,
        metrics,
        threshold_sources=threshold_sources,
        oof_f1_scores=oof_f1_scores,
        pipe=pipe,
    )

    report_path = REPORTS_DIR / BEST_MODEL_REPORT
    report_path.write_text(format_best_model_report(info) + "\n", encoding="utf-8")

    meta_path = MODELS_DIR / "metadata.joblib"
    if meta_path.exists():
        meta = joblib.load(meta_path)
    else:
        meta = {}
    meta["best_model"] = info
    joblib.dump(meta, meta_path)

    if verbose:
        print_best_model_summary(info)
        print(f"Zapisano: {best_path}")
        print(f"Raport:   {report_path}")

    return info


def load_best_model():
    """Wczytaj zapisany najlepszy model."""
    path = MODELS_DIR / BEST_MODEL_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Brak {path}. Uruchom: python scripts/train.py lub python scripts/run_train.py"
        )
    return joblib.load(path)


def load_best_model_info() -> dict:
    """Metadane najlepszego modelu z metadata.joblib lub raportu."""
    meta_path = MODELS_DIR / "metadata.joblib"
    if meta_path.is_file():
        meta = joblib.load(meta_path)
        if meta.get("best_model"):
            return meta["best_model"]
    report_path = REPORTS_DIR / BEST_MODEL_REPORT
    if report_path.is_file():
        return {"report_file": str(report_path)}
    return {}
