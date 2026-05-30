"""Metryki, porownanie modeli i feature importance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import REPORTS_DIR
from features import get_feature_names


def evaluate_classification(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Metryki klasyfikacji binarnej."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def evaluate_models(pipelines: dict, X_test, y_test) -> pd.DataFrame:
    """Porownaj wszystkie modele na zbiorze testowym."""
    rows = []
    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate_classification(y_test, y_pred, y_proba)
        metrics["model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("model")


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_classification_report_dict(y_true, y_pred) -> dict:
    return classification_report(y_true, y_pred, output_dict=True)


def _get_inner_pipeline(pipe):
    """Pipeline wewnetrzny: ThresholdClassifier → (CalibratedClassifierCV →) ImbPipeline."""
    inner = getattr(pipe, "pipeline", pipe)
    # CalibratedClassifierCV: weź pipeline z pierwszego skalibrowango modelu
    if hasattr(inner, "calibrated_classifiers_"):
        first = inner.calibrated_classifiers_[0]
        inner = getattr(first, "estimator", getattr(first, "base_estimator", first))
    return inner


def _importance_from_logistic(pipe, feature_names: list[str]) -> pd.Series:
    try:
        inner = _get_inner_pipeline(pipe)
        coefs = np.abs(inner.named_steps["model"].coef_[0])
        return pd.Series(coefs, index=feature_names).sort_values(ascending=False)
    except AttributeError:
        return pd.Series(0.0, index=feature_names)


def _importance_from_forest(pipe, feature_names: list[str]) -> pd.Series:
    inner = _get_inner_pipeline(pipe)
    imp = inner.named_steps["model"].feature_importances_
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def _importance_from_mlp(pipe, X_test, y_test, feature_names: list[str], random_state: int = 42) -> pd.Series:
    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        scoring="f1",
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)


def compute_feature_importance(
    pipelines: dict,
    preprocessor,
    num_cols: list[str],
    cat_cols: list[str],
    X_test,
    y_test,
) -> dict[str, pd.Series]:
    """Feature importance dla LR, RF i MLP."""
    encoded_names = get_feature_names(preprocessor, num_cols, cat_cols)
    raw_names = num_cols + cat_cols
    importance = {}

    if "logistic_regression" in pipelines:
        importance["logistic_regression"] = _importance_from_logistic(
            pipelines["logistic_regression"], encoded_names
        )
    for tree_name in ("random_forest", "balanced_rf", "xgboost"):
        if tree_name in pipelines:
            try:
                importance[tree_name] = _importance_from_forest(
                    pipelines[tree_name], encoded_names
                )
            except Exception:
                pass
    if "catboost" in pipelines:
        try:
            inner = _get_inner_pipeline(pipelines["catboost"])
            cb_model = inner.named_steps["model"]
            feat_names = list(cb_model.num_cols or []) + list(cb_model.cat_cols or [])
            importance["catboost"] = pd.Series(
                cb_model.feature_importances_, index=feat_names
            ).sort_values(ascending=False)
        except Exception:
            pass
    # Permutation importance wymaga tylko wybranych kolumn (pipeline ignoruje pozostale)
    X_test_sel = X_test[raw_names] if hasattr(X_test, "__getitem__") and hasattr(X_test, "columns") else X_test
    if "neural_network" in pipelines:
        importance["neural_network"] = _importance_from_mlp(
            pipelines["neural_network"], X_test_sel, y_test, raw_names
        )
    if "svm" in pipelines:
        importance["svm"] = _importance_from_mlp(
            pipelines["svm"], X_test_sel, y_test, raw_names
        )
    return importance


def plot_feature_importance(importance: dict[str, pd.Series], top_n: int = 15) -> list[Path]:
    """Zapisz wykresy feature importance do reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    sns.set_theme(style="whitegrid")

    for model_name, series in importance.items():
        top = series.head(top_n).sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        top.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Feature importance — {model_name.replace('_', ' ').title()} (top {top_n})")
        ax.set_xlabel("Wartosc waznosci")
        fig.tight_layout()
        path = REPORTS_DIR / f"feature_importance_{model_name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_mlp_learning_curve(
    pipeline,
    X_train,
    y_train,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Path:
    """Krzywa uczenia się MLP — F1 (train vs. CV) w funkcji liczby próbek treningowych."""
    from sklearn.model_selection import StratifiedKFold, learning_curve

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_sizes = np.linspace(0.1, 1.0, 8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring="f1",
        n_jobs=n_jobs,
        shuffle=True,
        random_state=random_state,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes_abs, train_mean, "o-", color="steelblue", label="Trening (F1)")
    ax.fill_between(
        train_sizes_abs,
        np.clip(train_mean - train_std, 0, 1),
        np.clip(train_mean + train_std, 0, 1),
        alpha=0.15,
        color="steelblue",
    )
    ax.plot(train_sizes_abs, val_mean, "s--", color="crimson", label="Walidacja CV (F1)")
    ax.fill_between(
        train_sizes_abs,
        np.clip(val_mean - val_std, 0, 1),
        np.clip(val_mean + val_std, 0, 1),
        alpha=0.15,
        color="crimson",
    )
    ax.set_xlabel("Liczba próbek treningowych")
    ax.set_ylabel("F1")
    ax.set_title("Krzywa uczenia się — MLP (sklearn)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    sns.despine(fig)
    fig.tight_layout()

    path = REPORTS_DIR / "mlp_learning_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix_heatmap(
    y_true,
    y_pred,
    title: str,
    path: Path,
) -> np.ndarray:
    """Heatmapa macierzy pomylek [[TN, FP], [FN, TP]]."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Prognoza: Zostaje", "Prognoza: Rezygnacja"],
        yticklabels=["Rzeczywistość: Zostaje", "Rzeczywistość: Rezygnacja"],
    )
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return cm


def _save_confusion_matrix_csv(cm: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(
        cm,
        index=["Rzeczywistość: Zostaje (0)", "Rzeczywistość: Rezygnacja (1)"],
        columns=["Prognoza: Zostaje (0)", "Prognoza: Rezygnacja (1)"],
    )
    df.to_csv(path)


def plot_confusion_matrices_for_test(
    pipelines: dict,
    X_test,
    y_test,
    *,
    best_key: str | None = None,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    Zapisuje:
    - reports/confusion_matrix_best.png (+ .csv) — model z najlepszym F1
    - reports/confusion_matrices.png — siatka wszystkich modeli
    """
    from best_model import display_name, select_best_model_key

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    if metrics_df is None:
        metrics_df = evaluate_models(pipelines, X_test, y_test)
    if best_key is None:
        best_key = select_best_model_key(metrics_df)

    if best_key in pipelines:
        y_pred = pipelines[best_key].predict(X_test)
        title = f"Macierz pomyłek — {display_name(best_key)} (zbiór testowy 20%)"
        best_png = REPORTS_DIR / "confusion_matrix_best.png"
        cm = plot_confusion_matrix_heatmap(y_test, y_pred, title, best_png)
        _save_confusion_matrix_csv(cm, REPORTS_DIR / "confusion_matrix_best.csv")
        out["best"] = best_png

    order = ["baseline"] + [k for k in pipelines if k != "baseline"]
    keys = order
    n = len(keys)
    if n == 0:
        return out

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, key in zip(axes_flat, keys):
        y_pred = pipelines[key].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["P0", "P1"],
            yticklabels=["R0", "R1"],
        )
        ax.set_title(display_name(key), fontsize=9)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Macierze pomyłek — zbiór testowy (holdout 20%)", fontsize=13, y=1.02)
    fig.tight_layout()
    grid_path = REPORTS_DIR / "confusion_matrices.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["all"] = grid_path
    return out


def plot_model_comparison(metrics_df: pd.DataFrame) -> Path:
    """Wykres porownania metryk modeli."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = metrics_df.reset_index().melt(id_vars="model", var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model", ax=ax)
    ax.set_title("Porownanie modeli (zbior testowy)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = REPORTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def summarize_cv_results(searches: dict) -> pd.DataFrame:
    """Podsumowanie najlepszych wynikow 5-fold CV (GridSearchCV)."""
    rows = []
    for name, search in searches.items():
        rows.append(
            {
                "model": name,
                "best_cv_f1": search.best_score_,
                "best_params": str(search.best_params_),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def save_reports(
    metrics_df: pd.DataFrame,
    importance: dict[str, pd.Series],
    cv_summary: pd.DataFrame | None = None,
) -> None:
    """Zapisz tabele CSV z metrykami i importance."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv")
    if cv_summary is not None:
        cv_summary.to_csv(REPORTS_DIR / "cv_summary.csv")
    for name, series in importance.items():
        series.to_csv(REPORTS_DIR / f"feature_importance_{name}.csv", header=["importance"])
