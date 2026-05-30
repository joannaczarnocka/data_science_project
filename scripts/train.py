"""Skrypt trenujacy modele z optymalizacja F1."""

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from best_model import save_best_model
from config import F1_TARGET
from evaluate import (
    compute_feature_importance,
    evaluate_models,
    plot_confusion_matrices_for_test,
    plot_feature_importance,
    plot_model_comparison,
    save_reports,
    summarize_cv_results,
)
from model import train_all_models


def _print_metrics_table(metrics: pd.DataFrame, label: str, f1_target: float) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"METRYKI — {label}")
    print(sep)
    print(f"{'Model':<30} {'F1':>8} {'Prec':>8} {'Recall':>8} {'ROC-AUC':>8} {'Acc':>8}")
    print("-" * 70)
    for name in metrics["f1"].sort_values(ascending=False).index:
        row = metrics.loc[name]
        ok = " *" if row["f1"] >= f1_target else ""
        print(
            f"{name:<30} {row['f1']:>8.4f} {row['precision']:>8.4f} "
            f"{row['recall']:>8.4f} {row.get('roc_auc', float('nan')):>8.4f} "
            f"{row.get('accuracy', float('nan')):>8.4f}{ok}"
        )
    print("-" * 70)
    print("  * = F1 >= cel\n")


def _print_test_validation(
    metrics_test: pd.DataFrame,
    metrics_train: pd.DataFrame,
    pipelines: dict,
    X_test,
    y_test,
    oof_f1_scores: dict,
    threshold_sources: dict,
) -> None:
    n = len(y_test)
    n_pos = int((y_test == 1).sum())
    print(f"\nProbki testowe: {n}  |  Attrition=1: {n_pos} ({n_pos / n:.1%})  |  cel F1: {F1_TARGET}")

    _print_metrics_table(metrics_train, "ZBIÓR TRENINGOWY (train_fit)", F1_TARGET)
    _print_metrics_table(metrics_test, "ZBIÓR TESTOWY (holdout 40%)", F1_TARGET)

    print("=" * 70)
    print("OVERFITTING -- roznica F1 (train - test):  >0.10 = mozliwy overfitting")
    print("-" * 70)
    for name in metrics_test.index:
        if name not in metrics_train.index:
            continue
        delta = metrics_train.loc[name, "f1"] - metrics_test.loc[name, "f1"]
        flag = " !" if delta > 0.10 else ""
        print(f"  {name:<30} d={delta:+.4f}{flag}")

    if oof_f1_scores:
        print("\nF1@prog (tuning OOF) vs F1 na TEST:")
        print(f"{'Model':<30} {'F1@prog':>10} {'zrodlo':>12} {'F1 test':>10}")
        print("-" * 54)
        for name in metrics_test.index:
            tune = oof_f1_scores.get(name, float("nan"))
            src = threshold_sources.get(name, "—")
            print(f"{name:<30} {tune:>10.4f} {src:>12} {metrics_test.loc[name, 'f1']:>10.4f}")

    best = metrics_test["f1"].idxmax()
    best_f1 = metrics_test["f1"].max()
    print(f"\nNajlepszy model na TEST: {best}  (F1={best_f1:.4f})")
    if best_f1 < F1_TARGET:
        print(
            f"UWAGA: F1={best_f1:.3f} < cel {F1_TARGET} — "
            "typowe przy ~16% Attrition."
        )

    print(f"\n--- Classification report: {best} ---")
    y_pred = pipelines[best].predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Zostaje (0)", "Rezygnacja (1)"], digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Macierz pomylek [[TN, FP], [FN, TP]]:")
    print(cm)
    if hasattr(pipelines[best], "threshold"):
        print(f"Prog decyzyjny: {pipelines[best].threshold:.4f}")
    print()


def main() -> None:
    print(f"Trenowanie (cel F1 >= {F1_TARGET}) — moze potrwac kilka minut...\n")
    artifacts = train_all_models(verbose=True)
    pipelines = artifacts["pipelines"]
    searches = artifacts["searches"]

    cv_summary = summarize_cv_results(searches)
    print("\nWyniki CV (Grid/RandomSearch, F1):")
    print(cv_summary.round(4).to_string())

    print(f"\nF1 przy dobranym progu (cel >= {F1_TARGET}, zrodlo: oof):")
    for name, score in artifacts["oof_f1_scores"].items():
        src = artifacts.get("threshold_sources", {}).get(name, "?")
        print(f"  {name}: {score:.4f}  [{src}]")

    metrics_test = evaluate_models(pipelines, artifacts["X_test"], artifacts["y_test"])
    metrics_train = evaluate_models(pipelines, artifacts["X_train"], artifacts["y_train"])

    importance = compute_feature_importance(
        pipelines,
        artifacts["preprocessor_fitted"],
        artifacts["num_cols"],
        artifacts["cat_cols"],
        artifacts["X_test"],
        artifacts["y_test"],
    )
    save_reports(metrics_test, importance, cv_summary)
    plot_feature_importance(importance)
    plot_model_comparison(metrics_test)
    plot_confusion_matrices_for_test(
        pipelines, artifacts["X_test"], artifacts["y_test"], metrics_df=metrics_test
    )

    save_best_model(
        pipelines,
        metrics_test,
        threshold_sources=artifacts.get("threshold_sources"),
        oof_f1_scores=artifacts.get("oof_f1_scores"),
        verbose=True,
    )
    print("\nZapisano: models/ (w tym best_model.joblib), reports/")

    _print_test_validation(
        metrics_test,
        metrics_train,
        pipelines,
        artifacts["X_test"],
        artifacts["y_test"],
        artifacts.get("oof_f1_scores", {}),
        artifacts.get("threshold_sources", {}),
    )


if __name__ == "__main__":
    main()
