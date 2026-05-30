"""
Clustering pracownikow HR — KMeans, opisy klastrów w jezyku naturalnym.
Wyniki: reports/clustering_*.png, reports/cluster_profiles.csv, reports/cluster_descriptions.txt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from data_loader import load_hr
from features import engineer_features

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# ── 1. Dane ───────────────────────────────────────────────────────────────────

print("Ladowanie i feature engineering...", flush=True)
df = engineer_features(load_hr())

# Cechy do clusteringu — interpretowalne, numeryczne
CLUSTER_FEATURES = [
    # Kariera i wynagrodzenie
    "Age", "MonthlyIncome", "JobLevel", "TotalWorkingYears",
    # Staz i stabilnosc
    "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager",
    "YearsSinceLastPromotion",
    # Satysfakcja
    "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
    "RelationshipSatisfaction", "AvgSatisfaction",
    # Mobilnosc i praca
    "NumCompaniesWorked", "HighOvertime", "DistanceFromHome",
    # Feature engineering
    "BurnoutRiskScore", "StabilityComposite", "CareerGrowthIndex",
    "CompaniesPerYear", "IncomePerJobLevel",
]

X_raw = df[CLUSTER_FEATURES].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── 2. Liczba klastrów — metoda łokcia + silhouette ───────────────────────────

print("Szukam optymalnego k...", flush=True)
ks = range(2, 8)
inertias, sil_scores = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels, sample_size=1000, random_state=42))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(list(ks), inertias, "o-", color="steelblue")
ax1.set_xlabel("Liczba klastrów k")
ax1.set_ylabel("Inercja (WCSS)")
ax1.set_title("Metoda łokcia")
ax2.plot(list(ks), sil_scores, "s-", color="crimson")
ax2.set_xlabel("Liczba klastrów k")
ax2.set_ylabel("Silhouette score")
ax2.set_title("Silhouette score")
fig.suptitle("Dobór liczby klastrów — pracownicy HR", fontsize=12)
fig.tight_layout()
fig.savefig(REPORTS / "clustering_elbow.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"  Silhouette scores: {dict(zip(ks, [round(s,3) for s in sil_scores]))}")

# Wymuszamy k=4 — interpretowalne klastry biznesowe
# (silhouette dla k=2 jest wyższe, ale 2 klastry dają za mało insights)
best_k = 4
print(f"  Wybrany k = {best_k} (wymuszony dla interpretowalności biznesowej)", flush=True)

# ── 3. KMeans z wybranym k ────────────────────────────────────────────────────

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["Cluster"] = km_final.fit_predict(X)

print(f"\nRozkład klastrów:")
for c, n in df["Cluster"].value_counts().sort_index().items():
    attr_rate = df[df["Cluster"] == c]["Attrition"].mean()
    print(f"  Klaster {c}: {n} pracowników ({n/len(df):.0%}), "
          f"Attrition={attr_rate:.1%}", flush=True)

# ── 4. Profile klastrów (srednie cech) ───────────────────────────────────────

profile_cols = [
    "Age", "MonthlyIncome", "JobLevel", "TotalWorkingYears",
    "YearsAtCompany", "YearsSinceLastPromotion",
    "JobSatisfaction", "WorkLifeBalance", "AvgSatisfaction",
    "NumCompaniesWorked", "HighOvertime", "DistanceFromHome",
    "BurnoutRiskScore", "StabilityComposite", "CareerGrowthIndex",
    "Attrition",
]
profiles = df.groupby("Cluster")[profile_cols].mean().round(2)
profiles["n"] = df["Cluster"].value_counts().sort_index()
global_mean = df[profile_cols].mean()
profiles.to_csv(REPORTS / "cluster_profiles.csv")
print("\nProfile klastrów zapisane.")

# ── 5. Wizualizacja PCA 2D ────────────────────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var = pca.explained_variance_ratio_

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
cluster_colors = [COLORS[c] for c in df["Cluster"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# PCA scatter — klastry
ax = axes[0]
for c in sorted(df["Cluster"].unique()):
    mask = df["Cluster"] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLORS[c], alpha=0.45, s=18, label=f"Klaster {c}")
ax.set_xlabel(f"PC1 ({var[0]:.1%} wariancji)")
ax.set_ylabel(f"PC2 ({var[1]:.1%} wariancji)")
ax.set_title("Klastry pracowników (PCA 2D)")
ax.legend(fontsize=9)

# PCA scatter — Attrition
ax = axes[1]
colors_attr = ["#66c2a5" if a == 0 else "#fc8d62" for a in df["Attrition"]]
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_attr, alpha=0.35, s=18)
p0 = mpatches.Patch(color="#66c2a5", label="Zostaje (0)")
p1 = mpatches.Patch(color="#fc8d62", label="Rezygnacja (1)")
ax.set_xlabel(f"PC1 ({var[0]:.1%} wariancji)")
ax.set_ylabel(f"PC2 ({var[1]:.1%} wariancji)")
ax.set_title("PCA 2D — Attrition (0/1)")
ax.legend(handles=[p0, p1], fontsize=9)

fig.suptitle("Clustering pracowników HR — PCA", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(REPORTS / "clustering_pca.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ── 6. Heatmapa profili ───────────────────────────────────────────────────────

heat_cols = [
    "Age", "MonthlyIncome", "JobLevel", "TotalWorkingYears",
    "YearsAtCompany", "YearsSinceLastPromotion",
    "JobSatisfaction", "WorkLifeBalance",
    "NumCompaniesWorked", "HighOvertime", "BurnoutRiskScore",
    "Attrition",
]
# Z-score profili względem globalnej średniej
profiles_z = (profiles[heat_cols] - global_mean[heat_cols]) / df[heat_cols].std()
fig, ax = plt.subplots(figsize=(13, 3.5))
sns.heatmap(
    profiles_z[heat_cols],
    annot=profiles[heat_cols].round(2),
    fmt=".2f",
    cmap="RdYlGn_r",
    center=0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Z-score vs. średnia"},
    annot_kws={"size": 8},
)
ax.set_title("Profile klastrów — Z-score względem średniej (czerwony=powyżej, zielony=poniżej)", pad=10)
ax.set_xlabel("")
ax.set_ylabel("Klaster")
fig.tight_layout()
fig.savefig(REPORTS / "clustering_heatmap.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ── 7. Opisy w języku naturalnym ─────────────────────────────────────────────

def describe_cluster(c_id, row, glob, n, attrition_rate):
    """Generuje opis klastru na podstawie odchyleń od średniej."""

    def cmp(col, high_label, low_label, threshold=0.15):
        val, avg = row[col], glob[col]
        diff = (val - avg) / (avg + 1e-6)
        if diff > threshold:
            return high_label
        if diff < -threshold:
            return low_label
        return None

    lines = [f"KLASTER {c_id} — {n} pracowników ({n/len(df):.0%} zespołu), "
             f"Attrition={attrition_rate:.1%}\n"]

    # Doswiadczenie i poziom
    age_lbl    = cmp("Age",             "starsi (śr. wiek: {:.0f} lat)", "młodsi (śr. wiek: {:.0f} lat)")
    jlvl_lbl   = cmp("JobLevel",        "wyższy poziom stanowiska", "niższy poziom stanowiska")
    income_lbl = cmp("MonthlyIncome",   "ponadprzeciętne wynagrodzenie (${:.0f}/mies.)",
                                        "poniżej przeciętnego wynagrodzenia (${:.0f}/mies.)")
    exp_lbl    = cmp("TotalWorkingYears","duże doświadczenie ({:.0f} lat)", "małe doświadczenie ({:.0f} lat)")

    # Stabilnosc w firmie
    ten_lbl    = cmp("YearsAtCompany",  "długi staż w firmie ({:.0f} lat)", "krótki staż w firmie ({:.0f} lat)")
    promo_lbl  = cmp("YearsSinceLastPromotion", "dawno bez awansu ({:.0f} lat)", "niedawno awansowany ({:.0f} lat)")

    # Satysfakcja i wellbeing
    sat_lbl    = cmp("AvgSatisfaction", "wysoka satysfakcja ({:.2f}/4)", "niska satysfakcja ({:.2f}/4)", 0.07)
    wlb_lbl    = cmp("WorkLifeBalance", "dobry work-life balance", "słaby work-life balance", 0.07)
    burn_lbl   = cmp("BurnoutRiskScore","wysoki wskaźnik wypalenia ({:.1f})", "niski wskaźnik wypalenia ({:.1f})", 0.15)

    # Mobilnosc
    mob_lbl    = cmp("NumCompaniesWorked","wysoka mobilność ({:.0f} firm)", "niska mobilność ({:.0f} firm)")
    ot_lbl     = cmp("HighOvertime",    "częste nadgodziny ({:.0%} grupy)", "rzadkie nadgodziny ({:.0%} grupy)", 0.2)

    def fmt(template, col):
        return template.format(row[col]) if template else None

    descs = []
    if age_lbl:    descs.append(fmt(age_lbl, "Age"))
    if exp_lbl:    descs.append(fmt(exp_lbl, "TotalWorkingYears"))
    if jlvl_lbl:   descs.append(jlvl_lbl)
    if income_lbl: descs.append(fmt(income_lbl, "MonthlyIncome"))
    if ten_lbl:    descs.append(fmt(ten_lbl, "YearsAtCompany"))
    if promo_lbl:  descs.append(fmt(promo_lbl, "YearsSinceLastPromotion"))
    if sat_lbl:    descs.append(fmt(sat_lbl, "AvgSatisfaction"))
    if wlb_lbl:    descs.append(wlb_lbl)
    if burn_lbl:   descs.append(fmt(burn_lbl, "BurnoutRiskScore"))
    if mob_lbl:    descs.append(fmt(mob_lbl, "NumCompaniesWorked"))
    if ot_lbl:     descs.append(fmt(ot_lbl, "HighOvertime"))

    lines.append("Charakterystyka:")
    for d in descs:
        lines.append(f"  - {d}")

    # Ryzyko attrition
    if attrition_rate > 0.25:
        lines.append(f"\n[!] WYSOKIE RYZYKO ODEJSCIA ({attrition_rate:.0%}) — priorytetowa interwencja HR")
    elif attrition_rate > 0.15:
        lines.append(f"\n[~] Podwyzszone ryzyko odejscia ({attrition_rate:.0%}) — monitorowanie zalecane")
    else:
        lines.append(f"\n[OK] Niskie ryzyko odejscia ({attrition_rate:.0%})")

    return "\n".join(lines)


# Generuj opisy
description_lines = [
    "=" * 65,
    "CLUSTERING PRACOWNIKÓW HR — OPISY KLASTRÓW",
    f"Metoda: KMeans (k={best_k}), StandardScaler, {len(CLUSTER_FEATURES)} cech",
    f"Silhouette score: {sil_scores[list(ks).index(best_k)]:.3f}",
    "=" * 65,
    "",
]

CLUSTER_NAMES = []
for c_id in sorted(df["Cluster"].unique()):
    row = profiles.loc[c_id]
    n = int(profiles.loc[c_id, "n"])
    attr_rate = row["Attrition"]
    desc = describe_cluster(c_id, row, global_mean, n, attr_rate)
    description_lines.append(desc)
    description_lines.append("")
    print(f"\n{desc}", flush=True)

    # Nadaj nazwy klastrów na podstawie profilu
    if attr_rate > 0.25:
        name = f"Klaster {c_id}: Ryzykowni"
    elif row["MonthlyIncome"] > global_mean["MonthlyIncome"] * 1.3 and row["TotalWorkingYears"] > global_mean["TotalWorkingYears"] * 1.2:
        name = f"Klaster {c_id}: Doświadczeni seniorzy"
    elif row["Age"] < global_mean["Age"] * 0.9 and row["NumCompaniesWorked"] > global_mean["NumCompaniesWorked"] * 1.2:
        name = f"Klaster {c_id}: Młodzi mobilni"
    elif row["AvgSatisfaction"] > global_mean["AvgSatisfaction"] * 1.05 and row["YearsAtCompany"] > global_mean["YearsAtCompany"]:
        name = f"Klaster {c_id}: Zaangażowani stali"
    else:
        name = f"Klaster {c_id}: Przeciętni stabilni"
    CLUSTER_NAMES.append(name)

description_lines += [
    "=" * 65,
    "NAZWY KLASTRÓW:",
    *[f"  {n}" for n in CLUSTER_NAMES],
    "=" * 65,
]

txt = "\n".join(description_lines)
(REPORTS / "cluster_descriptions.txt").write_text(txt, encoding="utf-8")
print(f"\nZapisano: {REPORTS / 'cluster_descriptions.txt'}")
print(f"Zapisano: {REPORTS / 'cluster_profiles.csv'}")
print(f"Wykresy:  clustering_elbow.png, clustering_pca.png, clustering_heatmap.png")

# ── 8. Wykres słupkowy: attrition wg klastra ──────────────────────────────────

cluster_stats = df.groupby("Cluster").agg(
    n=("Attrition", "count"),
    attrition_rate=("Attrition", "mean"),
    avg_income=("MonthlyIncome", "mean"),
    avg_age=("Age", "mean"),
    avg_satisfaction=("AvgSatisfaction", "mean"),
    avg_overtime=("HighOvertime", "mean"),
).round(3)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
x = [str(c) for c in cluster_stats.index]
bar_colors = [COLORS[c] for c in cluster_stats.index]

axes[0].bar(x, cluster_stats["attrition_rate"], color=bar_colors, edgecolor="white")
axes[0].axhline(df["Attrition"].mean(), color="crimson", ls="--", lw=1.5, label=f"Średnia={df['Attrition'].mean():.1%}")
axes[0].set_title("Attrition rate wg klastra")
axes[0].set_ylabel("P(Attrition=1)")
axes[0].set_ylim(0, min(1, cluster_stats["attrition_rate"].max() * 1.4))
axes[0].legend(fontsize=8)
for i, (c, v) in enumerate(zip(x, cluster_stats["attrition_rate"])):
    axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")

axes[1].bar(x, cluster_stats["avg_income"], color=bar_colors, edgecolor="white")
axes[1].axhline(df["MonthlyIncome"].mean(), color="crimson", ls="--", lw=1.5)
axes[1].set_title("Średnie wynagrodzenie ($/mies.)")
axes[1].set_ylabel("MonthlyIncome")
for i, (c, v) in enumerate(zip(x, cluster_stats["avg_income"])):
    axes[1].text(i, v + 50, f"${v:.0f}", ha="center", fontsize=8)

axes[2].bar(x, cluster_stats["avg_satisfaction"], color=bar_colors, edgecolor="white")
axes[2].axhline(df["AvgSatisfaction"].mean(), color="crimson", ls="--", lw=1.5)
axes[2].set_title("Średnia satysfakcja (1–4)")
axes[2].set_ylabel("AvgSatisfaction")
axes[2].set_ylim(0, 4)
for i, (c, v) in enumerate(zip(x, cluster_stats["avg_satisfaction"])):
    axes[2].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)

fig.suptitle(f"Statystyki klastrów pracowników (KMeans k={best_k})", fontsize=12)
fig.tight_layout()
fig.savefig(REPORTS / "clustering_stats.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Zapisano: clustering_stats.png")
print("\nClustering zakończony!")
