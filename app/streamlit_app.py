"""Streamlit — prognoza rezygnacji pracownika (HR Attrition)."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def _venv_python() -> Path:
    if sys.platform == "win32":
        return ROOT / "venv" / "Scripts" / "python.exe"
    return ROOT / "venv" / "bin" / "python"


VENV_PYTHON = _venv_python()
sys.path.insert(0, str(ROOT / "src"))

from best_model import BEST_MODEL_FILE, load_best_model_info
from config import CATEGORICAL_COLS, ENGINEERED_NUM_COLS, MODELS_DIR, REPORTS_DIR
from features import cast_categorical_columns, clean_data, engineer_features, get_X_y

# Zaleznosci wymagane do odpakowania zapisanych modeli (.joblib)
OPTIONAL_DEPS = {
    "xgboost":     "xgboost",
    "tensorflow":  "tensorflow",
    "catboost":    "catboost",
}


def _check_runtime_deps() -> dict[str, bool]:
    """Czy moduly do ladowania modeli sa zainstalowane w biezacym interpreterze."""
    status = {}
    for mod, pip_name in OPTIONAL_DEPS.items():
        try:
            __import__(mod)
            status[pip_name] = True
        except ImportError:
            status[pip_name] = False
    return status

st.set_page_config(page_title="HR Attrition Predictor", page_icon="📊", layout="wide")

MODEL_FILES = {
    # ensemble — top performery
    "Voting LR+SVM (najlepszy)":          "voting_lr_svm.joblib",
    "Voting LR+SVM+MLP":                  "voting.joblib",
    "Stacking (LR+SVM → meta-LR)":        "stacking.joblib",
    # bazowe
    "Logistic Regression":                "logistic_regression.joblib",
    "SVM":                                "svm.joblib",
    "Neural Network (MLP sklearn)":       "neural_network.joblib",
    "TensorFlow Neural Network":          "tf_neural_network.joblib",
    "XGBoost (early stop)":               "xgboost.joblib",
    "CatBoost":                           "catboost.joblib",
    "Balanced Random Forest":             "balanced_rf.joblib",
    "Random Forest":                      "random_forest.joblib",
    # warianty top-20 (wybrana selekcja cech per model)
    "Voting LR+SVM+MLP — top-20":         "voting_top10.joblib",
    "Balanced RF — top-20":               "balanced_rf_top10.joblib",
    "Random Forest — top-20":             "random_forest_top10.joblib",
    "LR — top-20":                        "logistic_regression_top10.joblib",
    "SVM — top-20":                       "svm_top10.joblib",
    "MLP — top-20":                       "neural_network_top10.joblib",
    "XGBoost — top-20":                   "xgboost_top10.joblib",
    "CatBoost — top-20":                  "catboost_top10.joblib",
    "TF Neural Net — top-20":             "tf_neural_network_top10.joblib",
    # punkt odniesienia
    "Baseline (najczęstsza klasa)":       "baseline.joblib",
}

INTEGER_LIKE = {
    "Age", "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TrainingTimesLastYear",
    "WorkLifeBalance", "NumCompaniesWorked", "PercentSalaryHike",
    "DistanceFromHome", "DailyRate", "HourlyRate", "MonthlyRate",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "TotalWorkingYears",
}


def _safe_load_model(path: Path, label: str) -> object | None:
    """Laduje model; przy braku zaleznosci zwraca None zamiast crashu."""
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        missing = exc.name or "nieznany"
        st.warning(
            f"Pomijam **{label}** — brak modulu `{missing}`. "
            f"Zainstaluj w tym samym Pythonie co Streamlit: "
            f"`pip install {missing}` lub uruchom: "
            f"`python scripts/run_streamlit.py`"
        )
        return None
    except Exception as exc:
        st.warning(f"Pomijam **{label}** — blad ladowania: {exc}")
        return None


@st.cache_resource
def load_artifacts():
    meta = joblib.load(MODELS_DIR / "metadata.joblib")
    best_info = meta.get("best_model") or load_best_model_info()

    best_pipe = None
    best_path = MODELS_DIR / BEST_MODEL_FILE
    if best_path.exists():
        best_label = best_info.get("display_name", "Najlepszy model")
        best_pipe = _safe_load_model(best_path, best_label)

    models = {}
    skipped = []
    for label, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if not path.exists():
            continue
        pipe = _safe_load_model(path, label)
        if pipe is not None:
            models[label] = pipe
        else:
            skipped.append(label)

    raw = pd.read_csv(ROOT / "data" / "raw" / "HR.csv")
    sample = clean_data(raw).iloc[0]
    return meta, best_pipe, best_info, models, sample, raw, skipped


def _is_integer_field(col: str, val) -> bool:
    if col.endswith("Level") or col in INTEGER_LIKE:
        return True
    return pd.api.types.is_integer_dtype(type(val)) or (
        isinstance(val, (int, float)) and float(val).is_integer()
    )


def _feature_columns_from_data(raw_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Pelny zestaw cech (num, cat, wszystkie) jak przy treningu."""
    _, _, full_num, full_cat = get_X_y(engineer_features(clean_data(raw_df)))
    form_num = [c for c in full_num if c not in ENGINEERED_NUM_COLS]
    all_cols = full_num + full_cat
    return form_num, full_cat, all_cols


def _map_importance_to_base(encoded_name: str) -> str:
    """Kolumna po OHE -> nazwa bazowa (np. OverTime_Yes -> OverTime)."""
    name = str(encoded_name)
    for base in CATEGORICAL_COLS:
        if name == base or name.startswith(f"{base}_"):
            return base
    return name


@st.cache_data
def _rf_importance_scores(form_columns: tuple[str, ...]) -> dict[str, float]:
    """Waznosc cech formularza wg Random Forest (max z kolumn OHE)."""
    path = REPORTS_DIR / "feature_importance_random_forest.csv"
    scores = {col: 0.0 for col in form_columns}
    if not path.exists():
        return scores

    series = pd.read_csv(path, index_col=0).iloc[:, 0].abs()
    for feat, imp in series.items():
        base = _map_importance_to_base(feat)
        if base in scores:
            scores[base] = max(scores[base], float(imp))
    return scores


def _sort_columns_by_rf_importance(columns: list[str]) -> list[str]:
    """Malejaco wg importance Random Forest."""
    scores = _rf_importance_scores(tuple(columns))
    return sorted(columns, key=lambda c: scores.get(c, 0.0), reverse=True)


def _unwrap_to_imb(inner) -> object | None:
    """Wyciaga pojedynczy ImbPipeline z ThresholdClassifier / SoftVotingEnsemble."""
    # SoftVotingEnsemble — ensembluje pipeline'y, wez pierwszego czlonka
    if hasattr(inner, "pipelines"):
        members = getattr(inner, "pipelines", [])
        return members[0] if members else None
    # ImbPipeline / sklearn Pipeline
    if hasattr(inner, "named_steps"):
        return inner
    return None


def _pipeline_input_columns(pipe) -> list[str]:
    """Kolumny oczekiwane przez pipeline na wejsciu (uwzglednia engineer/select/preprocess).

    Wszystkie pipeline'y zaczynaja od kroku `engineer` (FeatureEngineeringTransformer).
    Engineer jest idempotentny — akceptuje zarowno surowe jak i juz-engineered dane,
    o ile sa kolumny bazowe potrzebne do wyliczenia cech pochodnych.
    """
    inner = getattr(pipe, "pipeline", pipe)
    imb = _unwrap_to_imb(inner)
    if imb is None:
        return []
    steps = imb.named_steps

    # CatBoost: pipeline = [engineer, model], model.num_cols + model.cat_cols
    if "preprocess" not in steps and "model" in steps:
        model = steps["model"]
        num = list(getattr(model, "num_cols", []) or [])
        cat = list(getattr(model, "cat_cols", []) or [])
        return num + cat

    # Standardowo: preprocess (ColumnTransformer) ma feature_names_in_
    prep = steps.get("preprocess")
    if prep is None:
        return []
    if hasattr(prep, "feature_names_in_"):
        return list(prep.feature_names_in_)
    cols: list[str] = []
    for _, _, cols_in in prep.transformers_:
        if isinstance(cols_in, list):
            cols.extend(cols_in)
    return cols


def build_input_form(sample: pd.Series, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Formularz: surowe cechy (sortowane wg RF) -> feature engineering w tle."""
    form_num, form_cat, _ = _feature_columns_from_data(raw_df)
    form_cat_set = set(form_cat)
    form_fields = [c for c in form_num + form_cat if c in sample.index]
    ordered = _sort_columns_by_rf_importance(form_fields)
    row = {}

    st.sidebar.header("Dane pracownika")
    st.sidebar.caption("Kolejnosc pol: malejaca waznosc (Random Forest)")

    for col in ordered:
        if col in form_cat_set:
            options = sorted(raw_df[col].astype(str).unique())
            if col == "BusinessTravel":
                options = [o.replace("0n-Travel", "Non-Travel") for o in options]
            default = str(sample[col]).replace("0n-Travel", "Non-Travel")
            idx = options.index(default) if default in options else 0
            row[col] = st.sidebar.selectbox(col, options, index=idx)
            continue

        val = sample[col]
        if _is_integer_field(col, val):
            row[col] = st.sidebar.number_input(col, value=int(val), step=1)
        else:
            row[col] = st.sidebar.number_input(col, value=float(val), step=1.0)

    engineered = engineer_features(pd.DataFrame([row]))
    return cast_categorical_columns(engineered, list(form_cat_set))


def _pipeline_starts_with_engineer(pipe) -> bool:
    """Czy pipeline zaczyna sie od FeatureEngineeringTransformer."""
    inner = getattr(pipe, "pipeline", pipe)
    imb = _unwrap_to_imb(inner)
    if imb is None:
        return False
    steps = list(imb.named_steps.keys())
    return bool(steps) and steps[0] == "engineer"


def _align_X_for_pipeline(pipe, engineered: pd.DataFrame) -> pd.DataFrame:
    """Dopasuj wejscie do pipeline.

    Jesli pipeline ma `engineer` jako step 0 — przekazujemy CALE dane bez filtrowania
    kolumn. Engineer ponownie obliczy cechy pochodne (idempotentnie), nastepnie
    `select`/`preprocess` wybiora swoje kolumny po nazwie. To dziala uniwersalnie
    dla pipeline'ow pelnych, top-20 i CatBoost.

    Fallback dla starszych pipeline'ow bez `engineer`: kolumny po `preprocess`.
    """
    inner = getattr(pipe, "pipeline", pipe)
    imb = _unwrap_to_imb(inner)
    steps = list(imb.named_steps.keys()) if imb is not None else []
    starts_engineer = bool(steps) and steps[0] == "engineer"
    has_preprocess  = "preprocess" in steps
    has_select      = "select"     in steps

    # Pipeline z engineer + (select | preprocess): caly DataFrame — kroki obetna.
    if starts_engineer and (has_preprocess or has_select):
        return engineered.copy()

    # Pipeline z engineer ale tylko `[engineer, model]` (CatBoost full):
    # model oczekuje konkretnych kolumn w ustalonej kolejnosci.
    expected = _pipeline_input_columns(pipe)
    if not expected:
        return engineered.copy()
    missing = [c for c in expected if c not in engineered.columns]
    if missing:
        raise ValueError(
            f"Brakuje {len(missing)} cech w danych formularza: {missing[:5]}..."
        )
    return engineered.loc[:, expected].copy()


def main():
    st.title("HR Attrition — prognoza rezygnacji")

    deps = _check_runtime_deps()
    missing_deps = [name for name, ok in deps.items() if not ok]
    if missing_deps:
        st.error(
            "Brak pakietow w **tym** interpreterze Python: "
            + ", ".join(f"`{d}`" for d in missing_deps)
            + ". Streamlit czesto startuje z globalnego Pythona mimo aktywnego venv."
        )
        st.code("python scripts/run_streamlit.py", language="bash")

    if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        st.info(
            f"Aktywny Python: `{sys.executable}`\n\n"
            "Zalecane: `python scripts/run_streamlit.py`"
        )

    best_path = MODELS_DIR / BEST_MODEL_FILE
    if not best_path.exists() and not (MODELS_DIR / "xgboost.joblib").exists():
        st.error(
            "Brak modeli. Uruchom: `python scripts/train.py` "
            "lub `python scripts/update_best_model.py`"
        )
        st.stop()

    meta, best_pipe, best_info, models, sample, raw_df, skipped = load_artifacts()
    if best_pipe is None and not models:
        st.error(
            "Nie udalo sie zaladowac modelu. "
            "Zainstaluj zaleznosci (xgboost, catboost) w venv i uruchom ponownie."
        )
        st.stop()

    if best_info.get("display_name"):
        caption = (
            f"**Model produkcyjny:** {best_info['display_name']} "
            f"(`{best_info.get('model_key', 'best_model')}`) — "
            f"F1 test = {best_info.get('test_f1', 0):.3f}"
        )
        if best_info.get("test_precision") is not None:
            caption += (
                f", precision = {best_info['test_precision']:.3f}, "
                f"recall = {best_info['test_recall']:.3f}"
            )
        st.info(caption)

    if skipped:
        st.caption(f"Pominiete modele (porownanie): {', '.join(skipped)}")

    engineered = build_input_form(sample, raw_df)
    meta_cols = list(meta.get("num_cols", [])) + list(meta.get("cat_cols", []))
    _, _, all_cols = _feature_columns_from_data(raw_df)
    if len(meta_cols) < len(all_cols):
        st.warning(
            "Modele w `models/` zostaly wytrenowane na innym zestawie cech "
            f"({len(meta_cols)} vs {len(all_cols)}). Uruchom ponownie: `python scripts/train.py`"
        )

    if st.button("Prognozuj", type="primary"):
        st.session_state["predict"] = True

    if st.session_state.get("predict"):
        if best_pipe is not None:
            try:
                X_best = _align_X_for_pipeline(best_pipe, engineered)
                proba_best = best_pipe.predict_proba(X_best)[0]
                pred_best = int(best_pipe.predict(X_best)[0])
                label_best = best_info.get("display_name", "Najlepszy model")
                st.subheader(f"Prognoza — {label_best}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Decyzja", meta["class_labels"][pred_best])
                m2.metric("P(rezygnacja)", f"{proba_best[1]:.1%}")
                m3.metric("P(zostaje)", f"{proba_best[0]:.1%}")
                if hasattr(best_pipe, "threshold"):
                    st.caption(f"Prog decyzyjny: {best_pipe.threshold:.3f}")
            except ValueError as exc:
                st.error(f"Najlepszy model: {exc}")

        if models:
            with st.expander("Porownanie wszystkich modeli", expanded=best_pipe is None):
                results = []
                for label, pipe in models.items():
                    try:
                        X = _align_X_for_pipeline(pipe, engineered)
                    except ValueError as exc:
                        st.warning(f"{label}: {exc}")
                        continue
                    proba = pipe.predict_proba(X)[0]
                    pred = int(pipe.predict(X)[0])
                    results.append(
                        {
                            "Model": label,
                            "Prognoza": meta["class_labels"][pred],
                            "P(rezygnacja)": f"{proba[1]:.1%}",
                            "P(zostaje)": f"{proba[0]:.1%}",
                        }
                    )
                st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
                probs = []
                for pipe in models.values():
                    try:
                        Xp = _align_X_for_pipeline(pipe, engineered)
                        probs.append(pipe.predict_proba(Xp)[0][1])
                    except ValueError:
                        continue
                if probs:
                    st.metric(
                        f"Srednie P(rezygnacja) ({len(probs)} modele)",
                        f"{sum(probs) / len(probs):.1%}",
                    )

    st.divider()

    metrics_path = REPORTS_DIR / "model_metrics.csv"
    if metrics_path.exists():
        with st.expander("Wyniki modeli na zbiorze testowym (holdout 40%)", expanded=False):
            metrics_df = pd.read_csv(metrics_path, index_col=0)
            display_cols = [c for c in ["f1", "precision", "recall", "roc_auc", "accuracy"] if c in metrics_df.columns]
            if display_cols:
                df_show = metrics_df[display_cols].sort_values("f1", ascending=False).round(4)
                df_show.columns = [c.upper().replace("_", "-") for c in df_show.columns]
                st.dataframe(df_show, use_container_width=True)
                st.caption("Posortowane wg F1. 588 probek, ~16% Attrition (95 pozytywow).")

    st.caption("Raporty: folder `reports/`")


if __name__ == "__main__":
    if "predict" not in st.session_state:
        st.session_state["predict"] = False
    main()
