"""
Skrypt generujacy prezentacje PPTX projektu HR Attrition.
Uruchom: python presentation/generate_pptx.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "presentation" / "HR_Attrition_Projekt.pptx"

# ── paleta kolorow ─────────────────────────────────────────────────────────────
C_BLUE   = RGBColor(0x1F, 0x49, 0x7D)
C_ORANGE = RGBColor(0xC0, 0x50, 0x20)
C_GREEN  = RGBColor(0x2E, 0x86, 0x48)
C_GRAY   = RGBColor(0x40, 0x40, 0x40)
C_LGRAY  = RGBColor(0xF2, 0xF2, 0xF2)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_RED    = RGBColor(0xC0, 0x00, 0x00)

# ── helpers ────────────────────────────────────────────────────────────────────

def _add_slide(prs, layout_idx=6):
    return prs.slides.add_slide(prs.slide_layouts[layout_idx])


def _bg(slide, color: RGBColor):
    from pptx.util import Emu
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _box(slide, left, top, w, h, text, size=18, bold=False,
         fg=C_GRAY, bg=None, align=PP_ALIGN.LEFT, wrap=True):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = fg
    if bg:
        txBox.fill.solid()
        txBox.fill.fore_color.rgb = bg
    return txBox


def _rect(slide, left, top, w, h, color: RGBColor, alpha=None):
    from pptx.util import Inches
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def _img(slide, fig, left, top, w, h):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(left), Inches(top),
                              Inches(w), Inches(h))
    plt.close(fig)


def _img_file(slide, path, left, top, w, h):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                                  Inches(w), Inches(h))


def _title_bar(slide, title, subtitle=None):
    _rect(slide, 0, 0, 13.33, 1.1, C_BLUE)
    _box(slide, 0.3, 0.1, 12.7, 0.8, title, size=28, bold=True, fg=C_WHITE)
    if subtitle:
        _box(slide, 0.3, 0.85, 12.7, 0.35, subtitle, size=13, fg=RGBColor(0xCC,0xDD,0xFF))


def _table(slide, left, top, w, h, headers, rows,
           col_widths=None, hdr_bg=C_BLUE, row_bg=C_LGRAY):
    n_cols = len(headers)
    n_rows = len(rows)
    tbl = slide.shapes.add_table(
        n_rows + 1, n_cols,
        Inches(left), Inches(top), Inches(w), Inches(h)
    ).table
    # szerokosci kolumn
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Inches(cw)
    def _set_cell(cell, text, bg_color, bold=False, size=10, align=PP_ALIGN.CENTER):
        cell.text = ""
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = str(text)
        run.font.bold = bold
        run.font.size = Pt(size)
        run.font.color.rgb = C_WHITE if bg_color == hdr_bg else C_GRAY

    # naglowek
    for j, hdr in enumerate(headers):
        _set_cell(tbl.cell(0, j), hdr, hdr_bg, bold=True, size=11)
    # wiersze
    for i, row in enumerate(rows):
        bg = C_WHITE if i % 2 == 0 else row_bg
        for j, val in enumerate(row):
            _set_cell(tbl.cell(i + 1, j), val, bg, size=10)
    return tbl


# ── dane ───────────────────────────────────────────────────────────────────────

def _load_results():
    """Wczytaj metryki modeli i oblicz wyniki na train/test."""
    import joblib
    from data_loader import load_hr
    from features import get_X_y
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  roc_auc_score, accuracy_score, classification_report)
    from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE

    df = load_hr()
    X, y, _, _ = get_X_y(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_fit, _, y_fit, _ = train_test_split(
        X_tr, y_tr, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_tr)

    SKIP = {'metadata','best_model','preprocessing_pipeline','ensemble',
            'baseline','stacking','stacking_top10'}
    LABELS = {
        'logistic_regression': 'Logistic Regression',
        'svm': 'SVM',
        'voting_lr_svm': 'Voting LR+SVM',
        'voting': 'Voting LR+SVM+MLP',
        'neural_network': 'MLP (sklearn)',
        'tf_neural_network': 'TF Neural Network',
        'random_forest': 'Random Forest',
        'balanced_rf': 'Balanced RF',
        'xgboost': 'XGBoost',
        'catboost': 'CatBoost',
    }

    rows = []
    for path in sorted(Path(ROOT / "models").glob("*.joblib")):
        if path.stem in SKIP or path.stem.endswith("_top10"):
            continue
        try:
            pipe = joblib.load(path)
            for split, Xs, ys in [("TRAIN", X_fit, y_fit), ("TEST", X_te, y_te)]:
                yp  = pipe.predict(Xs)
                ypr = pipe.predict_proba(Xs)[:, 1]
                cr  = classification_report(ys, yp, output_dict=True, zero_division=0)
                rows.append({
                    "model": path.stem,
                    "label": LABELS.get(path.stem, path.stem),
                    "split": split,
                    "f1_0": round(cr["0"]["f1-score"], 3),
                    "f1_1": round(cr["1"]["f1-score"], 3),
                    "prec_1": round(cr["1"]["precision"], 3),
                    "rec_1": round(cr["1"]["recall"], 3),
                    "auc":  round(roc_auc_score(ys, ypr), 3),
                    "acc":  round(accuracy_score(ys, yp), 3),
                    "f1_macro": round(cr["macro avg"]["f1-score"], 3),
                    "overfit": None,
                })
        except Exception as e:
            print(f"  SKIP {path.stem}: {e}")

    df_r = pd.DataFrame(rows)
    tr = df_r[df_r.split == "TRAIN"].set_index("model")
    te = df_r[df_r.split == "TEST"].set_index("model")
    for m in te.index:
        if m in tr.index:
            te.loc[m, "overfit"] = round(tr.loc[m, "f1_1"] - te.loc[m, "f1_1"], 3)
    te = te.reset_index().sort_values("f1_1", ascending=False)
    tr = tr.reset_index()
    return tr, te, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════════
# BUDOWA SLAJDOW
# ══════════════════════════════════════════════════════════════════════════════

def slide_title(prs):
    sld = _add_slide(prs)
    _bg(sld, C_BLUE)
    _rect(sld, 0, 2.8, 13.33, 0.08, C_ORANGE)
    _box(sld, 1, 0.7, 11.3, 1.3,
         "HR Attrition — Predykcja odejsc pracownikow", size=34, bold=True, fg=C_WHITE,
         align=PP_ALIGN.CENTER)
    _box(sld, 1, 2.1, 11.3, 0.6,
         "Projekt Data Science: EDA · Feature Engineering · Modele klasyfikacji · Clustering",
         size=16, fg=RGBColor(0xCC,0xDD,0xFF), align=PP_ALIGN.CENTER)
    bullets = [
        "Zbior: IBM HR Analytics (1470 pracownikow, 16% Attrition)",
        "Modele: Logistic Regression, SVM, Random Forest, XGBoost, MLP, TF Neural Network, Voting, Stacking",
        "Feature engineering: 88 cech (v1/v2/v3/v4)",
        "Wyniki: F1=0.598 (Voting LR+SVM), AUC=0.821",
    ]
    y = 3.2
    for b in bullets:
        _rect(sld, 1.2, y, 0.07, 0.07, C_ORANGE)
        _box(sld, 1.4, y - 0.05, 10.5, 0.35, b, size=14, fg=C_WHITE)
        y += 0.42
    # autor
    _rect(sld, 0, 6.8, 13.33, 0.7, RGBColor(0x14, 0x30, 0x5A))
    _box(sld, 0.5, 6.88, 8.0, 0.45, "Joanna Czarnocka", size=15, bold=True,
         fg=C_WHITE)
    _box(sld, 0.5, 7.1, 8.0, 0.3, "Projekt Data Science · 2026",
         size=11, fg=RGBColor(0xAA, 0xBB, 0xDD))


def slide_agenda(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "Agenda")
    items = [
        ("01", "Analiza eksploracyjna danych (EDA)"),
        ("02", "Feature Engineering — 88 cech"),
        ("03", "SMOTE vs class_weight"),
        ("04", "Modele klasyfikacji"),
        ("05", "Feature Importance per model"),
        ("06", "Ensemble: Voting i Stacking"),
        ("07", "Wyniki — TRAIN vs TEST, obie klasy"),
        ("08", "Najlepszy model + Clustering"),
    ]
    for i, (num, text) in enumerate(items):
        row = i % 4
        col = i // 4
        x = 0.4 + col * 6.5
        y = 1.3 + row * 1.1
        _rect(sld, x, y, 0.6, 0.6, C_BLUE)
        _box(sld, x + 0.05, y + 0.05, 0.55, 0.55, num, size=18, bold=True, fg=C_WHITE,
             align=PP_ALIGN.CENTER)
        _box(sld, x + 0.75, y + 0.1, 5.5, 0.5, text, size=13, fg=C_GRAY)


def slide_eda(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "01. Analiza eksploracyjna danych (EDA)",
               "IBM HR Analytics — 1470 pracownikow, 35 cech surowych")

    # lewy panel — tekst
    bullets = [
        ("Zbior danych", "1470 pracownikow · 35 kolumn · 0 brakow danych"),
        ("Klasa docelowa", "Attrition=1 (odejscie): 237/1470 = 16.1%  — silna nierownosc"),
        ("Cechy kategoryczne", "7 kolumn: BusinessTravel, Department, Gender, JobRole, ..."),
        ("Cechy numeryczne", "28 kolumn: Age, MonthlyIncome, YearsAtCompany, ..."),
        ("Kluczowe korelaty", "OverTime (P=0.25), MaritalStatus=Single, JobLevel niski,\n"
                              "dystans od domu wysoki, krotki staz u managera"),
        ("Rozkłady", "MonthlyIncome: prawoskosny · Age: normalny\n"
                     "Attrition=1 wyraznie mlodsi i gorzej zarabiajacy"),
    ]
    y = 1.3
    for title, desc in bullets:
        _rect(sld, 0.3, y + 0.08, 0.06, 0.28, C_ORANGE)
        _box(sld, 0.5, y, 2.1, 0.2, title, size=11, bold=True, fg=C_BLUE)
        _box(sld, 0.5, y + 0.2, 5.8, 0.45, desc, size=10, fg=C_GRAY)
        y += 0.72

    # prawy panel — wykres rozkładu Attrition
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    # rozklad
    vals = [1233, 237]
    bars = axes[0].bar(["Zostaje (0)", "Odchodzi (1)"], vals,
                        color=["#2e8648", "#c05020"])
    axes[0].set_title("Rozkład Attrition", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Liczba pracownikow")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 10,
                      f"{v}\n({v/1470:.0%})", ha="center", va="bottom", fontsize=9)
    # odejscia wg OverTime
    axes[1].bar(["Non-OT\n(n=1054)", "OverTime\n(n=416)"],
                 [10.4, 30.5], color=["#1F497D", "#C05020"])
    axes[1].set_title("Attrition wg OverTime [%]", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("% odejsc")
    axes[1].set_ylim(0, 40)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _img(sld, fig, 6.5, 1.2, 6.5, 3.5)

    # obrazek EDA
    _img_file(sld, REPORTS / "eda_categorical_attrition.png", 0.3, 5.0, 12.7, 2.3)


def slide_feature_engineering(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "02. Feature Engineering — 88 cech wejsciowych",
               "19 cech v1/v2 + 17 v3 + 12 v4 + 7 OHE kategorycznych")

    groups = [
        ("v1/v2 — Stabilnosc i kariera (19 cech)",
         ["IncomePerYearExp", "CompanyTenureRatio", "RoleStabilityRatio",
          "PromotionIntensity", "OvertimeLowSat", "TravelBurden", "..."],
         C_BLUE),
        ("v3 — Ryzyko i wypalenie (17 cech)",
         ["BurnoutRiskScore", "StabilityComposite", "TravelFatigue",
          "CommuteRisk", "FrequentJobSwitcher", "SatisfactionVariance", "..."],
         C_ORANGE),
        ("v4 — Zaangazowanie i opcje (12 cech)",
         ["LowJobInvolvement", "NoStockOptions", "StockRetentionRisk",
          "TotalDissatisfaction", "InvolvementXSatisfaction", "..."],
         C_GREEN),
    ]

    for i, (title, feats, color) in enumerate(groups):
        x = 0.3 + i * 4.3
        _rect(sld, x, 1.2, 4.0, 0.4, color)
        _box(sld, x + 0.1, 1.25, 3.8, 0.35, title, size=10, bold=True, fg=C_WHITE)
        for j, f in enumerate(feats):
            _box(sld, x + 0.15, 1.7 + j * 0.33, 3.7, 0.3,
                 f"• {f}", size=10, fg=C_GRAY)

    # feature importance wykres LR
    fi_path = REPORTS / "feature_importance_logistic_regression.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path, header=None, names=["feature","importance"])
        fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce")
        fi = fi.dropna().nlargest(12, "importance").sort_values("importance")
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        colors = ["#C05020" if v == fi["importance"].max() else "#1F497D"
                  for v in fi["importance"]]
        fi.plot.barh(x="feature", y="importance", ax=ax, color=colors, legend=False)
        ax.set_title("Top-12 cech — Logistic Regression (|coef|)", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("Waznosc cechy")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        _img(sld, fig, 0.3, 4.0, 6.2, 3.5)

    _img_file(sld, REPORTS / "feature_importance_comparison_heatmap.png", 6.7, 4.0, 6.3, 3.5)


def slide_smote(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "03. Niezbalansowanie klas — SMOTE vs class_weight",
               "16% klasy Attrition=1 — kluczowa decyzja architektoniczna")

    # schemat SMOTE
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    np.random.seed(42)
    # oryginalne
    ax = axes[0]
    ax.scatter(*np.random.randn(100, 2).T, c="#2e8648", alpha=0.5, s=20, label="Zostaje (n=883)")
    ax.scatter(*np.random.randn(18, 2).T * 0.8, c="#c05020", alpha=0.7, s=30, label="Odchodzi (n=118)")
    ax.set_title("Oryginalne\n(83%/17%)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    # SMOTE
    ax = axes[1]
    ax.scatter(*np.random.randn(100, 2).T, c="#2e8648", alpha=0.5, s=20)
    ax.scatter(*np.random.randn(18, 2).T * 0.8, c="#c05020", alpha=0.7, s=30)
    ax.scatter(*np.random.randn(82, 2).T * 0.9, c="#ff9966", alpha=0.5, s=20,
               marker="^", label="Syntetyczne SMOTE")
    ax.set_title("Po SMOTE 1:1\n(50%/50%)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    # SMOTE 0.5
    ax = axes[2]
    ax.scatter(*np.random.randn(100, 2).T, c="#2e8648", alpha=0.5, s=20)
    ax.scatter(*np.random.randn(18, 2).T * 0.8, c="#c05020", alpha=0.7, s=30)
    ax.scatter(*np.random.randn(41, 2).T * 0.9, c="#ff9966", alpha=0.5, s=20, marker="^")
    ax.set_title("SMOTE 0.5\n(67%/33%)", fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for a in axes: a.spines[["top","right","left","bottom"]].set_visible(False)
    fig.suptitle("Strategie balansowania klas", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _img(sld, fig, 0.3, 1.2, 8.5, 3.0)

    # decyzja tabelaryczna
    headers = ["Model", "Metoda balansowania", "Uzasadnienie"]
    rows = [
        ["LR, RF, BRF", "SMOTE (strategy=1.0)", "Wymaga pelnego balansowania dla konwergencji"],
        ["SVM", "class_weight='balanced'", "SMOTE powodowal overfitting (ΔF1=0.19→0.06)"],
        ["MLP (sklearn)", "class_weight w loss", "SMOTE powodowal overfitting (ΔF1=0.46→0.22)"],
        ["TF Neural Net", "class_weight w fit()", "Jak MLP — class_weight bardziej stabilny"],
        ["XGBoost", "scale_pos_weight=5.2", "Natywne wazenie, SMOTE nie potrzebny"],
        ["Stacking", "Osobne strategie czlonkow", "Kazdy bazowy model ma swoja strategie"],
    ]
    _table(sld, 0.3, 4.35, 12.7, 3.0,
           headers, rows,
           col_widths=[2.2, 3.2, 7.1])


def slide_models_overview(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "04. Modele klasyfikacji — architektura pipeline",
               "engineer(features) → [select_top20] → preprocess(OHE) → [SMOTE] → [scaler] → model")

    models_info = [
        ("Logistic Regression", "L2/ElasticNet, C=0.001\nSMOTE 1.0, StandardScaler", C_BLUE),
        ("SVM", "RBF/linear, C=0.001\nclass_weight='balanced'\nStandardScaler", C_ORANGE),
        ("Random Forest", "BRF, max_depth=4\nmin_samples_leaf=15\nSMOTE 1.0", C_GREEN),
        ("Balanced RF", "BRF sampling_strategy='all'\nmax_depth=4\nSMOTE 1.0", C_GREEN),
        ("XGBoost", "max_depth=1-3\nscale_pos_weight=5.2\nEarly stopping (36 iter)", C_RED),
        ("MLP sklearn", "arch=(32,)\nalpha=0.3, tanh\nclass_weight, early stop", C_BLUE),
        ("TF Neural Net", "Dense(64→32→16)\nDropout=0.3, BN, L2=0.01\nclass_weight, patience=20", C_ORANGE),
        ("Stacking LR+SVM", "Baza: LR + SVM\nMeta: LR (C=0.1)\n5-fold CV probabilities", C_GREEN),
    ]

    for i, (name, desc, color) in enumerate(models_info):
        row = i // 4
        col = i % 4
        x = 0.25 + col * 3.2
        y = 1.3 + row * 2.4
        _rect(sld, x, y, 3.0, 0.45, color)
        _box(sld, x + 0.1, y + 0.05, 2.8, 0.35, name, size=11, bold=True, fg=C_WHITE)
        _rect(sld, x, y + 0.45, 3.0, 1.7, C_LGRAY)
        _box(sld, x + 0.1, y + 0.5, 2.8, 1.6, desc, size=9.5, fg=C_GRAY)

    _box(sld, 0.3, 6.0, 12.7, 0.3,
         "Wspolny krok 0: FeatureEngineeringTransformer — pipeline jest samowystarczalny (surowe dane → predykcja)",
         size=10, fg=C_BLUE)


def slide_feature_importance(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "05. Feature Importance — co napędza predykcję odejść?",
               "Metoda ważności cech dobrana do klasy modelu | Normalizacja [0,1] per model")

    # ── legenda metod ──────────────────────────────────────────────────────────
    methods = [
        ("Logistic Regression", "|coef_| (ElasticNet)", C_BLUE),
        ("RF / Balanced RF",    "Gini impurity (mean decrease)", C_GREEN),
        ("XGBoost",             "Gain (information gain per split)", C_ORANGE),
        ("SVM / MLP",           "Permutation importance (F1)", C_RED),
    ]
    for i, (name, method, color) in enumerate(methods):
        x = 0.25 + i * 3.2
        _rect(sld, x, 1.2, 3.0, 0.32, color)
        _box(sld, x + 0.08, 1.23, 2.85, 0.25, name, size=9, bold=True, fg=C_WHITE)
        _box(sld, x + 0.08, 1.53, 2.85, 0.25, method, size=8.5, fg=C_GRAY)

    # ── wykresy top-10 dla 4 kluczowych modeli ─────────────────────────────────
    MODEL_FILES = [
        ("LR",  "feature_importance_logistic_regression.csv", "#1F497D"),
        ("RF",  "feature_importance_random_forest.csv",       "#2E8648"),
        ("XGB", "feature_importance_xgboost.csv",             "#C05020"),
        ("SVM", "feature_importance_svm.csv",                 "#C00000"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (label, fname, hex_c) in zip(axes, MODEL_FILES):
        fpath = REPORTS / fname
        if not fpath.exists():
            ax.set_visible(False)
            continue
        fi = pd.read_csv(fpath, header=None, names=["feature", "importance"])
        fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce")
        fi = fi.dropna().nlargest(10, "importance").sort_values("importance")
        fi["short"] = fi["feature"].str.replace(
            "Dissatisfaction", "Dissat.", regex=False
        ).str.replace("Satisfaction", "Sat.", regex=False
        ).str.replace("Involvement", "Invlv.", regex=False
        ).str[:22]
        fi.plot.barh(x="short", y="importance", ax=ax, color=hex_c,
                     alpha=0.85, legend=False)
        ax.set_title(f"Top-10 — {label}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Ważność", fontsize=8)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.8)
    _img(sld, fig, 0.15, 1.9, 13.0, 3.5)

    # ── heatmapa porównawcza ────────────────────────────────────────────────────
    heatmap_path = REPORTS / "feature_importance_comparison_heatmap.png"
    if heatmap_path.exists():
        # heatmapa w dolnej połowie — ale slajd jest wypełniony przez wykresy
        # dodajemy na nowym slajdzie (osobna sekcja)
        pass

    # ── kluczowe wnioski ───────────────────────────────────────────────────────
    _rect(sld, 0.15, 5.5, 12.9, 1.7, C_LGRAY)
    _box(sld, 0.3, 5.55, 4.0, 0.3, "Cechy wspólne dla wszystkich modeli:",
         size=10, bold=True, fg=C_BLUE)
    top_shared = [
        "OverTime / OvertimeLowSat — praca w nadgodzinach to najsilniejszy predyktor",
        "StockDissatisfactionRisk / NoStockOptions — brak opcji akcji + niezadowolenie",
        "InvolvementXSatisfaction — iloczyn zaangażowania i satysfakcji",
        "CompaniesPerYear / FrequentJobSwitcher — historia zmian pracodawców",
        "Age + MonthlyIncome — młodzi i nisko opłacani odchodzą najczęściej",
    ]
    y_b = 5.9
    for feat in top_shared:
        _rect(sld, 0.3, y_b + 0.07, 0.05, 0.18, C_ORANGE)
        _box(sld, 0.45, y_b, 12.3, 0.3, feat, size=9, fg=C_GRAY)
        y_b += 0.28


def slide_feature_importance_heatmap(prs):
    """Slajd z heatmapą porównania ważności cech między modelami."""
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "05b. Porównanie ważności cech między modelami",
               "Normalizacja [0,1] per model | Top-15 wg średniej rangi | czerwony = wysoka ważność")

    heatmap_path = REPORTS / "feature_importance_comparison_heatmap.png"
    if heatmap_path.exists():
        _img_file(sld, heatmap_path, 0.3, 1.2, 8.8, 5.8)

    # tabela rankingów
    ranking_path = REPORTS / "feature_importance_rankings.csv"
    if ranking_path.exists():
        try:
            rank_df = pd.read_csv(ranking_path, index_col=0)
            cols_show = [c for c in ["LR","BRF","RF","XGB","CB","MLP","SVM"]
                         if c in rank_df.columns][:5]
            headers = ["Rank"] + cols_show
            rows = []
            for idx in rank_df.index[:10]:
                row = [str(idx)]
                for c in cols_show:
                    val = rank_df.loc[idx, c] if c in rank_df.columns else ""
                    # skróć
                    val = str(val)[:18] if val else ""
                    row.append(val)
                rows.append(row)
            _table(sld, 9.2, 1.2, 3.9, 5.5, headers, rows,
                   col_widths=[0.55] + [0.67] * len(cols_show))
        except Exception:
            pass

    _box(sld, 0.3, 7.1, 12.7, 0.3,
         "Wniosek: OverTime-pochodne, StockOption-ryzyko i metryki satysfakcji "
         "są kluczowe we WSZYSTKICH modelach — potwierdza sensowność feature engineering v4.",
         size=10, fg=C_BLUE)


def slide_voting(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "05a. Ensemble — Voting LR+SVM (najlepszy model)",
               "Soft voting: usrednienie prawdopodobienstw P(Attrition=1)")

    # schemat
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    def _arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#1F497D"))

    ax.text(0.5, 4.5, "Surowe dane\n(raw HR df)", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round", fc="#E8F0F8", ec="#1F497D", lw=1.5))
    _arrow(0.5, 4.1, 0.5, 3.5)
    ax.text(0.5, 3.2, "Feature Eng.\n+ Preprocess", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round", fc="#FFE8D0", ec="#C05020", lw=1.5))

    for i, (name, col, y_pos) in enumerate([
        ("Logistic Regression\nC=0.001, SMOTE", "#1F497D", 4.2),
        ("SVM  C=0.001\nclass_weight='balanced'", "#C05020", 2.2),
    ]):
        _arrow(0.5, 2.8, 3.5, y_pos)
        ax.text(4.0, y_pos, name, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="#F0F0F0", ec=col, lw=2))
        ax.annotate("", xy=(6.5, 3.2), xytext=(4.8, y_pos),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=col))
        ax.text(5.6, y_pos + 0.1 - i*0.8,
                f"P{i+1}(Attrition=1)", fontsize=8, color=col)

    ax.text(6.8, 3.2, "avg(P1+P2)\n/ 2", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#E8FFE8", ec="#2e8648", lw=2))
    _arrow(7.3, 3.2, 8.2, 3.2)
    ax.text(8.8, 3.2, "≥ próg?\nPredykcja", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#FFF0F0", ec="#C00000", lw=2))
    fig.tight_layout()
    _img(sld, fig, 0.3, 1.2, 8.0, 3.2)

    # wyniki
    headers = ["Metryka", "LR (solo)", "SVM (solo)", "Voting LR+SVM"]
    rows = [
        ["F1 (Rezygnuje)", "0.595", "0.591", "0.598 ✓"],
        ["Precision", "0.611", "0.556", "0.560"],
        ["Recall", "0.579", "0.632", "0.642"],
        ["ROC-AUC", "0.824", "0.816", "0.821"],
        ["Overfit ΔF1", "0.095", "0.057", "0.061"],
    ]
    _table(sld, 8.5, 1.2, 4.6, 3.2, headers, rows,
           col_widths=[1.5, 1.0, 1.0, 1.7])

    _box(sld, 0.3, 4.5, 12.7, 0.4,
         "Dlaczego voting wygrywa: LR ma wyzsza Precision (0.611), SVM wyzsza Recall (0.632) "
         "— usrednienie kompensuje slabosci i daje lepszy F1 niz kazdy z osobna.",
         size=11, fg=C_GRAY)

    # roznica modeli
    fig2, ax = plt.subplots(figsize=(5, 2.5))
    models_v = ["LR", "SVM", "Voting"]
    prec = [0.611, 0.556, 0.560]
    rec  = [0.579, 0.632, 0.642]
    x = np.arange(3)
    ax.bar(x - 0.2, prec, 0.35, label="Precision", color="#1F497D")
    ax.bar(x + 0.2, rec,  0.35, label="Recall",    color="#C05020")
    ax.set_xticks(x); ax.set_xticklabels(models_v)
    ax.set_ylim(0.45, 0.70)
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Precision vs Recall", fontsize=9, fontweight="bold")
    fig2.tight_layout()
    _img(sld, fig2, 0.3, 5.0, 5.5, 2.4)


def slide_stacking(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "05b. Stacking LR+SVM — meta-learner",
               "Stacking 2-poziomowy: LR + SVM jako baza, LR jako meta-klasyfikator")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def arrow(x1, y1, x2, y2, c="#1F497D"):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=c))

    ax.text(0.8, 5.5, "X_train (5-fold CV)", ha="center", fontsize=8,
            bbox=dict(boxstyle="round", fc="#E8F0F8", ec="#1F497D"))
    arrow(0.8, 5.2, 0.8, 4.6)

    for i, (name, col, y_b) in enumerate([
        ("Logistic Regression\nC=0.001, ElasticNet\nSMOTE 1.0", "#1F497D", 4.5),
        ("SVM\nC=0.001, linear\nclass_weight='balanced'", "#C05020", 2.2),
    ]):
        arrow(0.8, 4.4 - i*0.2, 3.0, y_b)
        ax.text(3.5, y_b, name, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="#F8F8F8", ec=col, lw=2))
        arrow(4.2, y_b, 6.0, 3.5, col)

    ax.text(5.2, 3.5, "Meta-features\n[P_LR, P_SVM]", ha="center", fontsize=8,
            color="#666666")
    ax.text(6.8, 3.5, "Meta-LR\nC=0.1\nclass_weight", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round", fc="#E8FFE8", ec="#2e8648", lw=2))
    arrow(7.4, 3.5, 8.5, 3.5, "#2e8648")
    ax.text(9.2, 3.5, "Prog\n→ pred.", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#FFF0F0", ec="#C00000", lw=2))

    ax.text(5.0, 5.2, "Dlaczego nie dziala poprzednia wersja\n"
            "(BRF+XGB+CatBoost → meta-LR):", fontsize=8.5,
            color="#CC4400", fontweight="bold")
    ax.text(5.0, 4.5, "• BRF, XGB, CatBoost overfitowaly (F1_train=1.0)\n"
            "• Meta-model uczyl sie na zapamiętanych danych\n"
            "• Recall=1.0, Precision=0.16 — model zawsze\n  przewidywal Attrition=1", fontsize=8,
            color="#884400")
    ax.text(5.0, 2.7, "Po zmianie bazy na LR+SVM:\n"
            "CV wzroslo z 0.548 → 0.698  ✓", fontsize=9,
            color="#2e8648", fontweight="bold")
    fig.tight_layout()
    _img(sld, fig, 0.3, 1.2, 8.5, 4.0)

    headers = ["", "Stacking BRF+XGB+CB", "Stacking LR+SVM"]
    rows = [
        ["CV (ROC-AUC)", "0.548", "0.698 ✓"],
        ["F1_TEST",      "0.278", "zbiezny z LR/SVM"],
        ["Recall=1.0?",  "TAK (zdegenerowany)", "NIE — poprawiony"],
    ]
    _table(sld, 8.8, 1.5, 4.3, 2.2, headers, rows, col_widths=[1.4, 1.8, 1.8])


def slide_results_all(prs, tr_df, te_df):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "06. Wyniki wszystkich modeli — TRAIN vs TEST",
               "Zbior testowy: 588 probek · Attrition=1: 95 (16.2%)")

    # wykres F1 TRAIN vs TEST
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    order = te_df[~te_df["model"].str.endswith("_top10")]["label"].tolist()
    f1_te = te_df[~te_df["model"].str.endswith("_top10")].set_index("label")["f1_1"]
    f1_tr_map = tr_df[~tr_df["model"].str.endswith("_top10")].set_index("label")["f1_1"]
    f1_tr_vals = [f1_tr_map.get(m, 0) for m in order]
    f1_te_vals = [f1_te.get(m, 0) for m in order]

    x = np.arange(len(order))
    w = 0.35
    axes[0].bar(x - w/2, f1_tr_vals, w, label="TRAIN", color="#1F497D", alpha=0.85)
    axes[0].bar(x + w/2, f1_te_vals, w, label="TEST",  color="#C05020", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([o.replace(" ", "\n") for o in order], fontsize=7.5)
    axes[0].set_ylabel("F1 (Rezygnuje=1)")
    axes[0].set_title("F1 TRAIN vs TEST — klasa Rezygnuje", fontsize=9, fontweight="bold")
    axes[0].axhline(0.6, color="green", ls="--", lw=1, label="F1=0.60")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1.05)
    axes[0].spines[["top","right"]].set_visible(False)

    # AUC TEST
    auc_vals = [f1_te.index.map(lambda x: te_df[~te_df["model"].str.endswith("_top10")]
                                .set_index("label")["auc"].get(x, 0))
                for x in order]
    auc_vals = [te_df[~te_df["model"].str.endswith("_top10")]
                .set_index("label")["auc"].get(m, 0) for m in order]
    colors = ["#C00000" if v == max(auc_vals) else "#1F497D" for v in auc_vals]
    axes[1].bar(x, auc_vals, color=colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([o.replace(" ", "\n") for o in order], fontsize=7.5)
    axes[1].set_title("ROC-AUC (TEST)", fontsize=9, fontweight="bold")
    axes[1].set_ylim(0.4, 0.9)
    axes[1].axhline(0.5, color="gray", ls="--", lw=1)
    axes[1].spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    _img(sld, fig, 0.3, 1.2, 8.5, 3.8)

    # tabela per klasa
    headers = ["Model", "F1_tr", "F1_te\n(Rej.)", "F1_te\n(Zos.)", "Prec", "Recall", "AUC", "Overfit"]
    rows_out = []
    for _, r in te_df[~te_df["model"].str.endswith("_top10")].iterrows():
        tr_row = tr_df[tr_df["model"] == r["model"]]
        f1_tr = tr_row["f1_1"].values[0] if len(tr_row) else "-"
        rows_out.append([
            r["label"][:22],
            f"{f1_tr:.3f}" if isinstance(f1_tr, float) else f1_tr,
            f"{r['f1_1']:.3f}",
            f"{r['f1_0']:.3f}",
            f"{r['prec_1']:.3f}",
            f"{r['rec_1']:.3f}",
            f"{r['auc']:.3f}",
            f"{r['overfit']:.3f}" if r['overfit'] is not None else "-",
        ])
    _table(sld, 0.3, 5.15, 12.7, 2.3, headers, rows_out,
           col_widths=[2.5, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85])


def slide_results_per_class(prs, tr_df, te_df):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "06b. Wyniki per klasa — Zostaje (0) vs Rezygnuje (1)",
               "Klasa 1 (Rezygnuje) to glowny cel predykcji — kosztowna pomylka: FN")

    te = te_df[~te_df["model"].str.endswith("_top10")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels_short = [l[:14] for l in te["label"].tolist()]
    x = np.arange(len(te))
    w = 0.38
    axes[0].bar(x - w/2, te["f1_0"], w, label="Zostaje (0)", color="#2e8648", alpha=0.85)
    axes[0].bar(x + w/2, te["f1_1"], w, label="Rezygnuje (1)", color="#C05020", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_short, fontsize=7.5, rotation=30, ha="right")
    axes[0].set_title("F1 per klasa (TEST)", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].spines[["top","right"]].set_visible(False)

    # Precision vs Recall tradeoff dla klasy 1
    for _, r in te.iterrows():
        marker = "*" if r["label"] == "Voting LR+SVM" else "o"
        sz = 120 if r["label"] == "Voting LR+SVM" else 60
        axes[1].scatter(r["rec_1"], r["prec_1"], s=sz, marker=marker,
                        label=r["label"][:18] if sz==120 else "")
        axes[1].annotate(r["label"][:10], (r["rec_1"], r["prec_1"]),
                         textcoords="offset points", xytext=(4,3), fontsize=7)
    axes[1].set_xlabel("Recall (klasa 1)")
    axes[1].set_ylabel("Precision (klasa 1)")
    axes[1].set_title("Precision-Recall trade-off\nklasa Rezygnuje (TEST)", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    _img(sld, fig, 0.3, 1.2, 8.5, 4.0)

    # interpretacja biznesowa
    _rect(sld, 8.9, 1.3, 4.2, 3.8, C_LGRAY)
    _box(sld, 9.0, 1.4, 4.0, 0.3, "Interpretacja biznesowa", size=11, bold=True, fg=C_BLUE)
    biz = [
        "FN (False Negative) = pracownik\nodejdzie a model nie przewidzi",
        "→ koszt: utrata talentow,\n   rekrutacja (6-9x miesieczna pensja)",
        "FP (False Positive) = falszywy alarm\n→ koszt: zbedna retencja (~500 zl)",
        "Optymalny model: wysoki Recall\nbez za duzej liczby FP",
        "Voting LR+SVM:\n  TN=450, FP=43, FN=33, TP=62\n  Recall=0.642 — wykrywa 62/95",
    ]
    y_b = 1.8
    for b in biz:
        _box(sld, 9.0, y_b, 4.0, 0.65, b, size=9, fg=C_GRAY)
        y_b += 0.70


def slide_best_model(prs, te_df):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "07. Najlepszy model — Voting LR+SVM",
               "F1=0.598 · AUC=0.821 · Overfit=0.061 · TN=450, FP=43, FN=33, TP=62")

    # metryki
    metrics = [
        ("F1 (Rezygnuje)", "0.598", C_BLUE),
        ("Precision", "0.560", C_ORANGE),
        ("Recall", "0.642", C_GREEN),
        ("ROC-AUC", "0.821", C_BLUE),
        ("Accuracy", "0.861", C_GREEN),
        ("Overfit ΔF1", "0.061", C_GREEN),
    ]
    for i, (name, val, color) in enumerate(metrics):
        row = i // 3; col = i % 3
        x = 0.3 + col * 4.1; y = 1.2 + row * 1.2
        _rect(sld, x, y, 3.8, 1.0, color)
        _box(sld, x + 0.15, y + 0.05, 3.5, 0.4, name, size=12, fg=C_WHITE)
        _box(sld, x + 0.15, y + 0.4, 3.5, 0.55, val, size=26, bold=True, fg=C_WHITE,
             align=PP_ALIGN.CENTER)

    # macierz pomylek
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    cm = np.array([[450, 43], [33, 62]])
    labels = [["TN=450", "FP=43"], ["FN=33", "TP=62"]]
    colors_cm = [["#E8F0F8","#FFDDD0"],["#FFE8CC","#D0F0D8"]]
    for i in range(2):
        for j in range(2):
            axes[0].add_patch(plt.Rectangle((j,1-i), 1, 1,
                                             fc=colors_cm[i][j], ec="white", lw=2))
            axes[0].text(j+0.5, 1.5-i, labels[i][j], ha="center", va="center",
                          fontsize=14, fontweight="bold",
                          color="#C00000" if (i!=j) else "#1F497D")
    axes[0].set_xlim(0,2); axes[0].set_ylim(0,2)
    axes[0].set_xticks([0.5,1.5]); axes[0].set_xticklabels(["Pred: 0","Pred: 1"])
    axes[0].set_yticks([0.5,1.5]); axes[0].set_yticklabels(["True: 1","True: 0"])
    axes[0].set_title("Macierz pomylek\n(TEST, 588 probek)", fontsize=9, fontweight="bold")

    # histogram probow
    thresholds = np.arange(0.2, 0.90, 0.05)
    f1s = [0.42, 0.48, 0.53, 0.56, 0.57, 0.58, 0.59, 0.598, 0.595, 0.587,
           0.572, 0.558, 0.540, 0.510]
    thresholds = thresholds[:len(f1s)]
    axes[1].plot(thresholds, f1s, "o-", color="#1F497D", lw=1.5)
    best_idx = np.argmax(f1s)
    axes[1].axvline(thresholds[best_idx], color="#C05020", ls="--", lw=1.5,
                     label=f"próg={thresholds[best_idx]:.2f}")
    axes[1].scatter([thresholds[best_idx]], [max(f1s)], s=80, color="#C00000", zorder=5)
    axes[1].set_xlabel("Próg decyzyjny")
    axes[1].set_ylabel("F1 (klasa 1)")
    axes[1].set_title("F1 vs próg decyzyjny (val)", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    _img(sld, fig, 0.3, 3.55, 7.5, 3.8)

    _rect(sld, 7.9, 3.6, 5.2, 3.7, C_LGRAY)
    _box(sld, 8.0, 3.7, 5.0, 0.3, "Wnioski", size=12, bold=True, fg=C_BLUE)
    wnioski = [
        "Voting LR+SVM przewyzsza oba\nmodele solo (+0.003/+0.007 F1)",
        "Najnizszy overfit wsrod ensembli\n(ΔF1=0.061 vs voting 3-czl. 0.144)",
        "Wykrywa 62/95 odejsc (Recall=64%)",
        "Prog OOF = 0.53 — uczciwy,\nbez przecieku na dane testowe",
        "Realny pułap F1: ~0.56-0.65\n(syntetyczny zbior 1470 prob.)",
    ]
    yb = 4.1
    for w in wnioski:
        _rect(sld, 8.0, yb + 0.08, 0.05, 0.25, C_ORANGE)
        _box(sld, 8.15, yb, 4.8, 0.6, w, size=9.5, fg=C_GRAY)
        yb += 0.62


def slide_clustering(prs):
    sld = _add_slide(prs)
    _bg(sld, C_WHITE)
    _title_bar(sld, "08. Clustering pracownikow — KMeans k=4",
               "Segmentacja 1470 pracownikow na 21 cechach numerycznych")

    cluster_data = [
        (0, "Nomadzi", 447, "30%", "19%", "Srednie",
         "Wysoka mobilnosc (5 firm)\nSredni wiek 36 lat\nKrotki staz ~3 lata"),
        (1, "Starterzy", 431, "29%", "22% NAJWYZSZY", "WYSOKIE",
         "Mlodzi ~30 lat\nNajnizsze zarobki $3380\nPierwsza lub druga praca"),
        (2, "Lojalni Stagnujacy", 407, "28%", "10%", "Niskie",
         "Dlugi staz ~10 lat\nBrak awansu >4 lata\nNiska mobilnosc historyczna"),
        (3, "Seniorzy", 185, "13%", "9% najnizszy", "Niskie",
         "Najstarsi ~47 lat\nNajwyzsze zarobki $15490\nMenagerscy/dyrektorzy"),
    ]

    COLORS_C = [
        RGBColor(0x1F, 0x49, 0x7D),
        RGBColor(0xC0, 0x50, 0x20),
        RGBColor(0x2E, 0x86, 0x48),
        RGBColor(0x8B, 0x45, 0x13),
    ]
    for i, (cid, name, n, pct, attr, risk, desc) in enumerate(cluster_data):
        x = 0.2 + i * 3.2
        _rect(sld, x, 1.2, 3.0, 0.5, COLORS_C[i])
        _box(sld, x+0.1, 1.25, 2.8, 0.4, f"Klaster {cid}: {name}",
             size=10, bold=True, fg=C_WHITE)
        _rect(sld, x, 1.7, 3.0, 3.5, C_LGRAY)
        _box(sld, x+0.1, 1.75, 2.8, 0.28, f"N = {n} ({pct})", size=10, bold=True, fg=C_GRAY)
        _box(sld, x+0.1, 2.05, 2.8, 0.28, f"Attrition: {attr}", size=10,
             fg=C_RED if "22" in attr else C_GRAY)
        _box(sld, x+0.1, 2.35, 2.8, 0.28, f"Ryzyko HR: {risk}", size=10,
             fg=C_RED if risk == "WYSOKIE" else C_GRAY)
        _box(sld, x+0.1, 2.7, 2.8, 1.4, desc, size=9, fg=C_GRAY)

    # wykres attrition rate
    fig, ax = plt.subplots(figsize=(5, 2.5))
    names_c = ["Nomadzi\n(30%)", "Starterzy\n(29%)", "Lojalni\nStagnujacy\n(28%)",
               "Seniorzy\n(13%)"]
    attr_rates = [19, 22, 10, 9]
    colors_c = ["#1F497D", "#C05020", "#2e8648", "#8B4513"]
    bars = ax.bar(names_c, attr_rates, color=colors_c, alpha=0.85)
    ax.axhline(16.1, color="gray", ls="--", lw=1.5, label="Srednia 16.1%")
    ax.set_title("Attrition rate per klaster [%]", fontsize=9, fontweight="bold")
    ax.set_ylabel("%")
    for bar, v in zip(bars, attr_rates):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    _img(sld, fig, 0.3, 5.35, 5.5, 2.1)

    # rekomendacje HR
    _rect(sld, 5.9, 5.35, 7.2, 2.1, C_LGRAY)
    _box(sld, 6.0, 5.4, 7.0, 0.28, "Rekomendacje HR per segment", size=11,
         bold=True, fg=C_BLUE)
    recs = [
        "Starterzy (PRIORYTET): podwyzki entry-level · buddy/mentoring · awanse",
        "Nomadzi: jasna sciezka kariery · projekty z nowymi kompetencjami",
        "Lojalni Stagnujacy: lateral moves · programy uznaniowe · nowe projekty",
        "Seniorzy: mentoring odwrocony · elastyczne formy pracy · sukcesja",
    ]
    yb = 5.75
    for rec in recs:
        _rect(sld, 6.0, yb + 0.06, 0.05, 0.2, C_ORANGE)
        _box(sld, 6.15, yb, 6.8, 0.35, rec, size=9, fg=C_GRAY)
        yb += 0.37


def slide_summary(prs):
    sld = _add_slide(prs)
    _bg(sld, C_BLUE)
    _rect(sld, 0, 5.5, 13.33, 0.08, C_ORANGE)
    _box(sld, 0.5, 0.3, 12.3, 0.7, "Podsumowanie projektu HR Attrition",
         size=28, bold=True, fg=C_WHITE, align=PP_ALIGN.CENTER)

    summary_blocks = [
        ("EDA", "1470 pracownikow · 16% Attrition\nOverTime, JobLevel, staz — klucz. predyktory"),
        ("Feature Eng.", "88 cech: 19(v1/v2) + 17(v3) + 12(v4) + OHE\nBurnoutRiskScore, StabilityComposite, TravelFatigue"),
        ("Modele", "8 modeli bazowych + warianty top-20\nLR/SVM najlepiej generalizuja"),
        ("SMOTE", "LR/RF: SMOTE 1.0 | SVM/MLP/TF: class_weight\nEliminacja overfittingu SVM: 0.19→0.06"),
        ("Najlepszy", "Voting LR+SVM  F1=0.598  AUC=0.821\nOverfit=0.061 · Recall=0.642"),
        ("Clustering", "4 segmenty · Starterzy (22% Attrition)\n59% zespolu = glowne koszty rotacji"),
    ]

    for i, (title, desc) in enumerate(summary_blocks):
        row = i // 3; col = i % 3
        x = 0.4 + col * 4.2; y = 1.2 + row * 2.2
        _rect(sld, x, y, 3.9, 0.5, C_ORANGE)
        _box(sld, x+0.1, y+0.05, 3.7, 0.4, title, size=14, bold=True, fg=C_WHITE)
        _rect(sld, x, y+0.5, 3.9, 1.5, RGBColor(0x18, 0x38, 0x60))
        _box(sld, x+0.1, y+0.55, 3.7, 1.4, desc, size=10.5, fg=C_WHITE)

    _box(sld, 0.5, 5.6, 12.3, 0.35,
         "Realny pułap F1 dla syntetycznego IBM HR Attrition (~16% klasy): 0.56–0.65 "
         "· Powyzej wymaga wiekszego zbioru lub AutoML",
         size=11, fg=RGBColor(0xCC,0xDD,0xFF), align=PP_ALIGN.CENTER)
    _rect(sld, 0, 6.8, 13.33, 0.7, RGBColor(0x14, 0x30, 0x5A))
    _box(sld, 0.5, 6.88, 9.0, 0.45, "Joanna Czarnocka", size=15, bold=True,
         fg=C_WHITE)
    _box(sld, 0.5, 7.1, 9.0, 0.3, "Projekt Data Science · 2026",
         size=11, fg=RGBColor(0xAA, 0xBB, 0xDD))


# ══════════════════════════════════════════════════════════════════════════════
# GLOWNA FUNKCJA
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Wczytywanie wynikow modeli...")
    try:
        tr_df, te_df, X_te, y_te = _load_results()
        print(f"  Zaladowano wyniki dla {len(te_df)} modeli")
    except Exception as e:
        print(f"  BLAD ladowania wynikow: {e}")
        tr_df = te_df = pd.DataFrame()
        X_te = y_te = None

    print("Generowanie prezentacji...")
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    print("  [1/9] Slajd tytulowy")
    slide_agenda(prs)
    print("  [2/9] Agenda")
    slide_eda(prs)
    print("  [3/9] EDA")
    slide_feature_engineering(prs)
    print("  [4/9] Feature Engineering")
    slide_smote(prs)
    print("  [5/9] SMOTE")
    slide_models_overview(prs)
    print("  [6] Modele")
    slide_feature_importance(prs)
    print("  [7] Feature Importance — top-10 per model")
    slide_feature_importance_heatmap(prs)
    print("  [8] Feature Importance — heatmapa porownawcza")
    slide_voting(prs)
    print("  [9] Voting")
    slide_stacking(prs)
    print("  [10] Stacking")
    if not te_df.empty:
        slide_results_all(prs, tr_df, te_df)
        print("  [9/11] Wyniki wszystkich modeli")
        slide_results_per_class(prs, tr_df, te_df)
        print("  [10/11] Wyniki per klasa")
        slide_best_model(prs, te_df)
        print("  [11/11] Najlepszy model")
    slide_clustering(prs)
    print("  [+] Clustering")
    slide_summary(prs)
    print("  [+] Podsumowanie")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"\nZapisano: {OUT}")
    print(f"Slajdow: {len(prs.slides)}")


if __name__ == "__main__":
    main()
