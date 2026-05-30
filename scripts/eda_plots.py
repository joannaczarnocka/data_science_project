"""Generuje wykresy EDA do reports/ (histogramy, korelacja, Attrition)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_hr
from features import clean_data

HIST_COLS = [
    "Age",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "DistanceFromHome",
    "JobSatisfaction",
]

CAT_COLS = ["OverTime", "Department", "JobRole", "MaritalStatus", "BusinessTravel"]

EDA_FILES = {
    "attrition": REPORTS / "eda_attrition_distribution.png",
    "histograms": REPORTS / "eda_histograms.png",
    "correlation": REPORTS / "eda_correlation_heatmap.png",
    "categorical": REPORTS / "eda_categorical_attrition.png",
    "boxplots": REPORTS / "eda_numeric_boxplots.png",
}


def _prepare_df() -> pd.DataFrame:
    df = clean_data(load_hr())
    if df["Attrition"].dtype == object:
        df["Attrition"] = (df["Attrition"].str.lower() == "yes").astype(int)
    return df


def compute_eda_summary(df: pd.DataFrame) -> dict:
    """Statystyki do slajdu podsumowującego EDA."""
    n = len(df)
    attr = int((df["Attrition"] == 1).sum())
    missing = int(df.isnull().sum().sum())

    overtime_rate = None
    if "OverTime" in df.columns:
        ot = df[df["OverTime"].astype(str).str.lower().isin(["yes", "1", "true"])]
        if len(ot):
            overtime_rate = float(ot["Attrition"].mean())

    return {
        "n_rows": n,
        "n_cols": df.shape[1],
        "attrition_pct": attr / n,
        "attrition_n": attr,
        "missing": missing,
        "overtime_attrition_rate": overtime_rate,
    }


def plot_attrition_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df["Attrition"].value_counts().sort_index()
    labels = ["Zostaje (0)", "Rezygnacja (1)"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#66c2a5", "#fc8d62"])
    ax.set_title("Rozkład zmiennej docelowej Attrition")
    ax.set_ylabel("Liczba pracowników")
    vals = [counts.get(0, 0), counts.get(1, 0)]
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, str(int(val)), ha="center", fontsize=10)
    pct = counts.get(1, 0) / len(df) * 100
    ax.text(0.98, 0.95, f"Rezygnacja: {pct:.1f}%", transform=ax.transAxes, ha="right", va="top", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, path: Path) -> None:
    """Histogramy cech numerycznych — osobno dla Attrition=0 i Attrition=1."""
    cols = [c for c in HIST_COLS if c in df.columns]
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        for label, subset, color in [
            (0, df[df["Attrition"] == 0][col], "#66c2a5"),
            (1, df[df["Attrition"] == 1][col], "#fc8d62"),
        ]:
            ax.hist(subset.dropna(), bins=25, alpha=0.55, label=f"Attrition={label}", color=color, edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Częstość")
        ax.legend(fontsize=7)

    for ax in axes[len(cols) :]:
        ax.axis("off")

    fig.suptitle("Histogramy cech numerycznych wg Attrition", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Mapa korelacji — cechy numeryczne (+ Attrition)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Attrition" not in num_cols and "Attrition" in df.columns:
        num_cols.append("Attrition")

    drop = [c for c in ("EmployeeCount", "StandardHours", "EmployeeNumber") if c in num_cols]
    num_cols = [c for c in num_cols if c not in drop]

    corr = df[num_cols].corr()

    # Kolejność: Attrition na górze korelacji z targetem
    if "Attrition" in corr.columns:
        order = corr["Attrition"].abs().sort_values(ascending=False).index.tolist()
        corr = corr.loc[order, order]

    fig_h = max(8, len(corr) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_h, fig_h * 0.92))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
        annot=len(corr) <= 20,
        fmt=".2f",
        annot_kws={"size": 6},
    )
    ax.set_title("Macierz korelacji — cechy numeryczne", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_categorical_attrition(df: pd.DataFrame, path: Path) -> None:
    cols = [c for c in CAT_COLS if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    for i, col in enumerate(cols):
        rates = df.groupby(col)["Attrition"].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=axes[i], color="#e76f51", edgecolor="white")
        axes[i].set_title(f"P(rezygnacja) wg {col}")
        axes[i].set_ylabel("P(Attrition=1)")
        axes[i].tick_params(axis="x", rotation=35, labelsize=8)
        axes[i].set_ylim(0, min(1.0, rates.max() * 1.25 + 0.05))
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Cechy kategoryczne a ryzyko rezygnacji", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_numeric_boxplots(df: pd.DataFrame, path: Path) -> None:
    key_num = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "DistanceFromHome"]
    key_num = [c for c in key_num if c in df.columns]
    fig, axes = plt.subplots(1, len(key_num), figsize=(14, 4))
    if len(key_num) == 1:
        axes = [axes]
    for ax, col in zip(axes, key_num):
        df.boxplot(column=col, by="Attrition", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("Attrition (0/1)")
    fig.suptitle("Rozkład cech numerycznych wg Attrition (boxplot)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_eda_plots(reports_dir: Path | None = None) -> tuple[dict, dict[str, Path]]:
    """Generuje wszystkie wykresy EDA. Zwraca (summary, ścieżki plików)."""
    out_dir = reports_dir or REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 100

    df = _prepare_df()
    summary = compute_eda_summary(df)

    paths = {
        "attrition": out_dir / "eda_attrition_distribution.png",
        "histograms": out_dir / "eda_histograms.png",
        "correlation": out_dir / "eda_correlation_heatmap.png",
        "categorical": out_dir / "eda_categorical_attrition.png",
        "boxplots": out_dir / "eda_numeric_boxplots.png",
    }

    plot_attrition_distribution(df, paths["attrition"])
    plot_histograms(df, paths["histograms"])
    plot_correlation_heatmap(df, paths["correlation"])
    plot_categorical_attrition(df, paths["categorical"])
    plot_numeric_boxplots(df, paths["boxplots"])

    return summary, paths


if __name__ == "__main__":
    summary, paths = generate_eda_plots()
    print("EDA summary:", summary)
    for name, p in paths.items():
        print(f"  {name}: {p}")
