"""Generuje szczegolowy raport per-model i per-klasa po treningu.

Uruchom po scripts/train.py:
    python scripts/save_model_reports.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE, VAL_SIZE
from data_loader import load_hr
from features import get_X_y

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "baseline":            "Baseline",
    "logistic_regression": "Logistic Regression",
    "balanced_rf":         "Balanced RF",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "catboost":            "CatBoost",
    "svm":                 "SVM",
    "neural_network":      "MLP (sklearn)",
    "stacking":            "Stacking (BRF+XGB+CB+SVM / LR)",
}

_SKIP = {"metadata", "best_model", "preprocessing_pipeline", "ensemble"}


def load_pipelines():
    pipelines = {}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if path.stem in _SKIP:
            continue
        pipelines[path.stem] = joblib.load(path)
    return pipelines


def per_class_rows(name, pipe, X_test, y_test):
    label = MODEL_LABELS.get(name, name)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    cr  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = round(roc_auc_score(y_test, y_proba), 4)
    thr = round(getattr(pipe, "threshold", 0.5), 4)

    rows = []
    for cls_key, cls_label in [
        ("0",          "Zostaje (0)"),
        ("1",          "Rezygnuje (1)"),
        ("macro avg",  "Macro avg"),
        ("weighted avg", "Weighted avg"),
    ]:
        rows.append({
            "model":     label,
            "klasa":     cls_label,
            "precision": round(cr[cls_key]["precision"], 4),
            "recall":    round(cr[cls_key]["recall"],    4),
            "f1":        round(cr[cls_key]["f1-score"],  4),
            "support":   int(cr[cls_key]["support"]),
            "roc_auc":   auc   if cls_key == "1" else None,
            "threshold": thr   if cls_key == "1" else None,
        })
    return rows


def confusion_text(name, pipe, X_test, y_test):
    label = MODEL_LABELS.get(name, name)
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return (
        f"{label}\n"
        f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}\n"
        f"  Precision={tp/(tp+fp+1e-9):.4f}  Recall={tp/(tp+fn+1e-9):.4f}\n"
    )


def main():
    print("Wczytywanie danych i modeli...")
    df = load_hr()
    X, y, _, _ = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipelines = load_pipelines()
    if not pipelines:
        print("Brak modeli w models/ — najpierw uruchom scripts/train.py")
        return

    print(f"Zaladowano {len(pipelines)} modeli: {list(pipelines.keys())}")
    print(f"Zbior testowy: {len(X_test)} probek, Attrition=1: {int((y_test==1).sum())}")

    # ── Per-class DataFrame ────────────────────────────────────────────────────
    rows = []
    for name, pipe in pipelines.items():
        rows.extend(per_class_rows(name, pipe, X_test, y_test))

    df_report = pd.DataFrame(rows)
    out_csv = REPORTS_DIR / "detailed_per_class_report.csv"
    df_report.to_csv(out_csv, index=False)
    print(f"\nZapisano: {out_csv}")

    # ── Tekstowy raport dla każdego modelu ────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("SZCZEGOLOWY RAPORT PER-MODEL — HR Attrition")
    lines.append(f"Zbior testowy: {len(X_test)} probek  |  Attrition=1: {int((y_test==1).sum())}")
    lines.append("=" * 70)

    resign_rows = df_report[df_report["klasa"] == "Rezygnuje (1)"].copy()
    resign_rows = resign_rows.sort_values("f1", ascending=False)

    lines.append("\n[KLASA 1 — REZYGNUJE]  (posortowane wg F1, support=95)")
    lines.append(f"{'Model':<32} {'F1':>7} {'Prec':>7} {'Recall':>7} {'ROC-AUC':>9} {'Prog':>7}")
    lines.append("-" * 70)
    for _, r in resign_rows.iterrows():
        lines.append(
            f"{r['model']:<32} {r['f1']:>7.4f} {r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} {r['roc_auc']:>9.4f} {r['threshold']:>7.4f}"
        )

    stay_rows = df_report[df_report["klasa"] == "Zostaje (0)"].copy()
    stay_rows = stay_rows.sort_values("f1", ascending=False)
    lines.append("\n[KLASA 0 — ZOSTAJE]  (support=493)")
    lines.append(f"{'Model':<32} {'F1':>7} {'Prec':>7} {'Recall':>7}")
    lines.append("-" * 55)
    for _, r in stay_rows.iterrows():
        lines.append(f"{r['model']:<32} {r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")

    macro_rows = df_report[df_report["klasa"] == "Macro avg"].copy()
    macro_rows = macro_rows.sort_values("f1", ascending=False)
    lines.append("\n[MACRO AVG — srednia klas 0 i 1]")
    lines.append(f"{'Model':<32} {'F1':>7} {'Prec':>7} {'Recall':>7}")
    lines.append("-" * 55)
    for _, r in macro_rows.iterrows():
        lines.append(f"{r['model']:<32} {r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")

    lines.append("\n" + "=" * 70)
    lines.append("MACIERZE POMYLEK")
    lines.append("=" * 70)
    for name, pipe in sorted(
        pipelines.items(),
        key=lambda kv: resign_rows[resign_rows["model"] == MODEL_LABELS.get(kv[0], kv[0])]["f1"].values[0]
        if MODEL_LABELS.get(kv[0], kv[0]) in resign_rows["model"].values else -1,
        reverse=True,
    ):
        lines.append(confusion_text(name, pipe, X_test, y_test))

    report_text = "\n".join(lines)
    out_txt = REPORTS_DIR / "detailed_model_report.txt"
    out_txt.write_text(report_text, encoding="utf-8")
    print(f"Zapisano: {out_txt}")

    # ── Drukuj na ekranie ─────────────────────────────────────────────────────
    print("\n" + report_text)


if __name__ == "__main__":
    main()
