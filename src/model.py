"""Trenowanie: SMOTE, OOF threshold, optymalizacja F1."""

from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Uśrednianie prawdopodobieństw z kilku dopasowanych pipeline'ów."""

    def __init__(self, pipelines: list, weights: list | None = None):
        self.pipelines = pipelines
        self.weights = weights

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        probas = [p.predict_proba(X) for p in self.pipelines]
        if self.weights:
            w = np.array(self.weights, dtype=float)
            w /= w.sum()
            return sum(wi * pi for wi, pi in zip(w, probas))
        return np.mean(probas, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    """CatBoost z natywną obsługą kategorycznych — DataFrame bezpośrednio do CatBoost."""

    def __init__(
        self,
        num_cols=None,
        cat_cols=None,
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        random_seed=42,
    ):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def _to_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        cols = list(self.num_cols or []) + list(self.cat_cols or [])
        return pd.DataFrame(X, columns=cols)

    def fit(self, X, y):
        df = self._to_df(X)
        num_cols = list(self.num_cols or [])
        cat_cols = list(self.cat_cols or [])
        if num_cols:
            self.num_imputer_ = SimpleImputer(strategy="median")
            df[num_cols] = self.num_imputer_.fit_transform(df[num_cols].astype(float))
        if cat_cols:
            self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = self.cat_imputer_.fit_transform(df[cat_cols].astype(str))
            for col in cat_cols:
                df[col] = df[col].astype(str)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=self.random_seed,
        )
        self.model_.fit(df, y, cat_features=cat_cols or None)
        self.classes_ = self.model_.classes_
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def _transform(self, X) -> pd.DataFrame:
        df = self._to_df(X)
        num_cols = list(self.num_cols or [])
        cat_cols = list(self.cat_cols or [])
        if num_cols and hasattr(self, "num_imputer_"):
            df[num_cols] = self.num_imputer_.transform(df[num_cols].astype(float))
        if cat_cols and hasattr(self, "cat_imputer_"):
            df[cat_cols] = self.cat_imputer_.transform(df[cat_cols].astype(str))
            for col in cat_cols:
                df[col] = df[col].astype(str)
        return df

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform(X))

    def predict(self, X):
        return self.model_.predict(self._transform(X))

from config import (
    CV_FOLDS,
    CV_SCORING,
    F1_TARGET,
    MAX_SELECTED_FEATURES_PER_MODEL,
    MODELS_DIR,
    RANDOM_SEARCH_ITER,
    RANDOM_STATE,
    SMOTE_STRATEGY,
    TEST_SIZE,
    VAL_SIZE,
)
from data_loader import load_hr, save_processed
from features import (
    ColumnSelector,
    FeatureEngineeringTransformer,
    build_preprocessor,
    engineer_features,
    fit_and_save_preprocessing_pipeline,
    get_X_y,
    get_feature_names,
    save_preprocessed_dataset,
)
from threshold import ThresholdClassifier, find_best_threshold, select_threshold
try:
    from tf_model import TFNeuralNetworkClassifier
    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False

TREE_MODELS = {"random_forest", "xgboost", "balanced_rf"}
SCALED_MODELS = {"logistic_regression", "neural_network", "svm", "tf_neural_network"}
NO_SMOTE_MODELS = {"stacking", "xgboost", "balanced_rf", "random_forest",
                   "neural_network", "svm", "tf_neural_network"}  # class_weight zamiast SMOTE
MLP_SEARCH_ITER = 15
NN_MODELS = ("neural_network",)


def get_base_estimators(
    scale_pos_weight: float,
    num_cols: list | None = None,
    cat_cols: list | None = None,
) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            C=0.01,
            class_weight={0: 1, 1: 3},
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            random_state=RANDOM_STATE,
        ),
        "balanced_rf": BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            sampling_strategy="all",
            replacement=True,
        ),
        "random_forest": BalancedRandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "neural_network": MLPClassifier(
            activation="relu",
            solver="adam",
            alpha=0.1,           # silna regularyzacja L2 (bylo 5e-3)
            hidden_layer_sizes=(32, 16),  # mniejsza architektura (bylo 128,64)
            batch_size=32,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,  # szybszy early stop (bylo 20)
            random_state=RANDOM_STATE,
        ),
        "svm": SVC(
            probability=True,
            class_weight="balanced",
            C=0.05,              # silniejsza regularyzacja (bylo 0.5)
            kernel="rbf",
            gamma="scale",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",  # precision-recall AUC — lepszy dla niezbalansowanych
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight * 1.5,
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.6,
            colsample_bytree=0.6,
            min_child_weight=10,
            reg_alpha=2.0,
            reg_lambda=5.0,
            gamma=1.0,
        ),
        **({
            "tf_neural_network": TFNeuralNetworkClassifier(
                hidden_units=(64, 32, 16),
                dropout_rate=0.3,
                learning_rate=1e-3,
                epochs=200,
                batch_size=32,
                patience=20,
                l2_reg=0.01,
                random_state=RANDOM_STATE,
            )
        } if _TF_AVAILABLE else {}),
        "stacking": StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(
                    max_iter=5000, C=0.01, solver="saga", penalty="elasticnet",
                    l1_ratio=0.5, class_weight={0: 1, 1: 3}, random_state=RANDOM_STATE,
                )),
                ("svm", SVC(
                    probability=True, class_weight="balanced",
                    C=0.05, kernel="rbf", gamma="scale", random_state=RANDOM_STATE,
                )),
            ],
            final_estimator=LogisticRegression(
                max_iter=5000, C=0.1, solver="lbfgs",
                class_weight={0: 1, 1: 3}, random_state=RANDOM_STATE,
            ),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            stack_method="predict_proba",
            n_jobs=1,
        ),
    }


def get_param_grids() -> dict[str, list | dict]:
    return {
        # saga solver supports l1, l2, elasticnet — search all three
        "logistic_regression": [
            {
                "model__penalty": ["l1", "l2"],
                "model__C": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            },
            {
                "model__penalty": ["elasticnet"],
                "model__C": [0.001, 0.005, 0.01, 0.05],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        ],
        "svm": {
            "model__C": [0.001, 0.01, 0.05, 0.1],  # mniejszy zakres, silniejsza reg.
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
    }


def get_param_distributions(scale_pos_weight: float) -> dict[str, dict]:
    return {
        "xgboost": {
            "model__n_estimators": [50, 75, 100],
            "model__max_depth": [1, 2, 3],         # include stumps (depth=1)
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.5, 0.6, 0.7],
            "model__colsample_bytree": [0.4, 0.5, 0.6],
            "model__min_child_weight": [10, 20, 30],
            "model__reg_alpha": [2.0, 5.0, 10.0],
            "model__reg_lambda": [5.0, 10.0, 20.0],
            "model__scale_pos_weight": [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2],
        },
        "balanced_rf": {
            "model__n_estimators": [200, 300],
            "model__max_depth": [3, 4, 6],
            "model__min_samples_leaf": [10, 20, 30],       # bylo 5-12, teraz agresywniej
            "model__max_features": ["sqrt", "log2"],
        },
        "random_forest": {
            "model__n_estimators": [200, 300],
            "model__max_depth": [3, 4, 6],
            "model__min_samples_leaf": [15, 25, 40],       # bylo 8-12
        },
        "neural_network": {
            "model__hidden_layer_sizes": [(32, 16), (64, 32), (32,)],   # mniejsze
            "model__alpha": [0.05, 0.1, 0.3, 0.5],        # silna reg (bylo 1e-3 - 1e-1)
            "model__learning_rate_init": [5e-4, 1e-3, 2e-3],
            "model__activation": ["relu", "tanh"],
        },
        "catboost": {
            "model__iterations": [200, 300, 400],          # bylo 600-1000
            "model__depth": [3, 4, 5],                     # bylo 4-6
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__l2_leaf_reg": [15, 20, 30, 50],        # bylo 3-12
        },
    }


def _build_baseline_pipeline(preprocessor) -> ImbPipeline:
    """Baseline: zawsze klasa wiekszosciowa (bez SMOTE, bez strojenia)."""
    return ImbPipeline(
        [
            ("engineer", FeatureEngineeringTransformer()),
            ("preprocess", preprocessor),
            (
                "model",
                DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
            ),
        ]
    )


def _build_pipeline(preprocessor, estimator, model_name: str) -> ImbPipeline:
    # CatBoostWrapper obsługuje preprocessing wewnętrznie — krok 0: inżynieria cech
    if model_name == "catboost":
        return ImbPipeline([("engineer", FeatureEngineeringTransformer()), ("model", estimator)])
    steps: list[tuple[str, object]] = [
        ("engineer", FeatureEngineeringTransformer()),
        ("preprocess", preprocessor),
    ]
    if model_name not in NO_SMOTE_MODELS:
        smote = SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=5,
            sampling_strategy=SMOTE_STRATEGY,
        )
        steps.append(("smote", smote))
    if model_name in SCALED_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


class _FitOnceResult:
    """Wynik pojedynczego fit — kompatybilny z train_all_models (bez wielokrotnego CV)."""

    def __init__(self, estimator, best_score: float = float("nan")):
        self.best_estimator_ = estimator
        self.best_score_ = best_score
        self.best_params_ = {
            k: v for k, v in estimator.get_params(deep=False).items() if k.startswith("model__")
        }
        self.cv_results_ = {
            "params": [self.best_params_],
            "mean_test_score": [best_score],
        }


def _fit_baseline(pipe, X_train, y_train) -> _FitOnceResult:
    """Dopasuj baseline i oszacuj F1 na CV (5-fold)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipe, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1
    )
    pipe.fit(X_train, y_train)
    return _FitOnceResult(pipe, best_score=float(scores.mean()))


def _run_search(pipe, model_name: str, X_train, y_train, scale_pos_weight: float):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if model_name in ("stacking", "tf_neural_network"):
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=1
        )
        pipe.fit(X_train, y_train)
        return _FitOnceResult(pipe, best_score=float(scores.mean()))

    dists = get_param_distributions(scale_pos_weight)

    if model_name in TREE_MODELS or model_name in NN_MODELS:
        n_iter = RANDOM_SEARCH_ITER
        n_jobs = -1
        if model_name == "neural_network":
            n_iter = MLP_SEARCH_ITER
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=dists[model_name],
            n_iter=n_iter,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=n_jobs,
            refit=True,
            random_state=RANDOM_STATE,
        ).fit(X_train, y_train)
    else:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=get_param_grids()[model_name],
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=-1,
            refit=True,
        ).fit(X_train, y_train)

    cv_scores = cross_val_score(
        search.best_estimator_, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=1
    )
    search.best_score_ = float(cv_scores.mean())

    return search


def _wrap_with_threshold(
    pipe, X_train, y_train, X_val=None, y_val=None,
) -> tuple[ThresholdClassifier, float, float, str]:
    """Prog pod F1>=F1_TARGET: OOF → val (uczciwy); bez train/smote_train."""
    threshold, tuning_f1, source = select_threshold(
        pipe, X_train, y_train, F1_TARGET, CV_FOLDS, RANDOM_STATE,
        X_val=X_val, y_val=y_val,
    )
    wrapped = ThresholdClassifier(pipe, threshold=threshold, threshold_source=source)
    return wrapped, threshold, tuning_f1, source



def _get_top_k_cols_for_model(
    model_name: str,
    tuned_pipe,
    num_cols: list[str],
    cat_cols: list[str],
    k: int = MAX_SELECTED_FEATURES_PER_MODEL,
    X_val=None,
    y_val=None,
) -> list[str]:
    """Top-k oryginalnych cech dla konkretnego modelu.

    Metoda dobierana do klasy modelu:
    - drzewa (RF, BRF, XGB): feature_importances_ (agregacja OHE→oryginalne)
    - LR: |coef_| (agregacja OHE→oryginalne)
    - CatBoost: feature_importances_ bezpośrednio w oryginalnej przestrzeni
    - SVM, MLP, Stacking: permutation importance (sub-pipeline bez kroków
      engineer/smote, X_val w przestrzeni inżynierowanych cech)
    """
    all_cols = num_cols + cat_cols
    fallback = all_cols[:k]

    def _fill(top: list[str]) -> list[str]:
        if len(top) < k:
            top = top + [c for c in all_cols if c not in set(top)][: k - len(top)]
        return top

    def _agg_ohe(imp_series: pd.Series) -> list[str]:
        """Agregacja OHE importancji → oryginalne kolumny, zwraca top-k."""
        orig: dict[str, float] = {}
        for feat, imp in imp_series.items():
            matched = False
            for cat_col in cat_cols:
                if feat.startswith(cat_col + "_"):
                    orig[cat_col] = orig.get(cat_col, 0.0) + float(imp)
                    matched = True
                    break
            if not matched:
                orig[feat] = orig.get(feat, 0.0) + float(imp)
        valid = set(all_cols)
        ranked = sorted(
            (c for c in orig if c in valid), key=lambda c: orig[c], reverse=True
        )[:k]
        return _fill(ranked)

    try:
        inner = tuned_pipe.pipeline          # ImbPipeline lub SoftVotingEnsemble
        steps  = getattr(inner, "named_steps", {})

        # ── RF, BRF, XGB — Gini/gain importances (OHE) ───────────────────
        if model_name in ("random_forest", "balanced_rf", "xgboost"):
            feat_names = get_feature_names(steps["preprocess"], num_cols, cat_cols)
            imp = steps["model"].feature_importances_
            return _agg_ohe(pd.Series(dict(zip(feat_names, imp))))

        # ── LR — |coef_| (OHE) ───────────────────────────────────────────
        if model_name == "logistic_regression":
            feat_names = get_feature_names(steps["preprocess"], num_cols, cat_cols)
            imp = np.abs(steps["model"].coef_[0])
            return _agg_ohe(pd.Series(dict(zip(feat_names, imp))))

        # ── CatBoost — importances w oryginalnej przestrzeni (bez OHE) ───
        if model_name == "catboost":
            cb = steps["model"]
            feat_names = list(cb.num_cols or []) + list(cb.cat_cols or [])
            imp = cb.feature_importances_
            return _agg_ohe(pd.Series(dict(zip(feat_names, imp))))

        # ── SVM, MLP, Stacking — permutation importance ───────────────────
        if X_val is None or y_val is None or not steps:
            return fallback

        # Sub-pipeline: pomijamy engineer + smote; X_val jest już inżynierowany
        sub = [(n, t) for n, t in inner.steps if n not in ("engineer", "smote")]
        if not sub:
            return fallback

        X_sel = X_val[all_cols] if hasattr(X_val, "columns") else X_val

        def _sub_predict(X_df):
            X_t = X_df
            for _, step in sub[:-1]:
                X_t = step.transform(X_t)
            proba = sub[-1][1].predict_proba(X_t)[:, 1]
            return (proba >= tuned_pipe.threshold).astype(int)

        base_f1 = f1_score(y_val, _sub_predict(X_sel), zero_division=0)
        rng = np.random.default_rng(RANDOM_STATE)
        imp_dict: dict[str, float] = {}
        for col in all_cols:
            drops = []
            for _ in range(5):
                X_perm = X_sel.copy()
                X_perm[col] = rng.permutation(X_perm[col].values)
                drops.append(base_f1 - f1_score(y_val, _sub_predict(X_perm), zero_division=0))
            imp_dict[col] = float(np.mean(drops))

        ranked = sorted(imp_dict, key=imp_dict.get, reverse=True)[:k]
        return _fill(ranked)

    except Exception:
        return fallback


def _build_pipeline_top10(
    preprocessor_top10,
    estimator,
    model_name: str,
    top10_cols: list[str],
) -> ImbPipeline:
    """Pipeline z selekcją cech: engineer → select_top10 → preprocess → ... → model."""
    selector = ColumnSelector(top10_cols)
    if model_name == "catboost":
        return ImbPipeline([
            ("engineer", FeatureEngineeringTransformer()),
            ("select", selector),
            ("model", estimator),
        ])
    steps: list[tuple[str, object]] = [
        ("engineer", FeatureEngineeringTransformer()),
        ("select", selector),
        ("preprocess", preprocessor_top10),
    ]
    if model_name not in NO_SMOTE_MODELS:
        steps.append(("smote", SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=5,
            sampling_strategy=SMOTE_STRATEGY,
        )))
    if model_name in SCALED_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def train_all_models(verbose: bool = True) -> dict:
    df = load_hr()
    save_processed(engineer_features(df), "hr_engineered.csv")
    if verbose:
        path = save_preprocessed_dataset(df)
        print(f"Zapisano zbior po preprocessingu: {path}", flush=True)

    X, y, num_cols, cat_cols = get_X_y(df)
    all_num_cols, all_cat_cols = num_cols, cat_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # Wydziel zbiór walidacyjny do doboru progu (model nie widzi X_val podczas fit)
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train,
    )
    if verbose:
        print(
            f"Podział danych: train_fit={len(X_train_fit)}, val={len(X_val)}, test={len(X_test)} "
            f"(razem {len(X)})",
            flush=True,
        )

    scale_pos_weight = float((y_train_fit == 0).sum() / max((y_train_fit == 1).sum(), 1))
    num_cols_used, cat_cols_used = num_cols, cat_cols

    fit_and_save_preprocessing_pipeline(X_train_fit, y_train_fit, num_cols_used, cat_cols_used)
    preprocessor = build_preprocessor(num_cols_used, cat_cols_used)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipelines = {}
    searches = {}
    best_params = {}
    thresholds = {}
    oof_f1_scores = {}
    threshold_sources = {}

    majority_class = int(pd.Series(y_train_fit).value_counts().idxmax())
    if verbose:
        label = "Zostaje (0)" if majority_class == 0 else "Rezygnacja (1)"
        print(
            f"  -> baseline (najczestsza klasa: {label})...",
            flush=True,
        )
    baseline_pipe = _build_baseline_pipeline(preprocessor)
    baseline_search = _fit_baseline(baseline_pipe, X_train_fit, y_train_fit)
    searches["baseline"] = baseline_search
    baseline_fitted = baseline_search.best_estimator_
    baseline_tuned = ThresholdClassifier(
        baseline_fitted,
        threshold=0.5,
        threshold_source="most_frequent",
    )
    baseline_f1 = float(
        f1_score(y_val, baseline_tuned.predict(X_val), zero_division=0)
    )
    baseline_threshold = 0.5
    baseline_src = "most_frequent"
    pipelines["baseline"] = baseline_tuned
    best_params["baseline"] = baseline_search.best_params_
    thresholds["baseline"] = baseline_threshold
    oof_f1_scores["baseline"] = baseline_f1
    threshold_sources["baseline"] = baseline_src
    joblib.dump(baseline_tuned, MODELS_DIR / "baseline.joblib")
    pd.DataFrame(baseline_search.cv_results_).to_csv(
        MODELS_DIR / "cv_results_baseline.csv", index=False
    )
    if verbose:
        mark = "OK" if baseline_f1 >= F1_TARGET else "ponizej celu"
        print(
            f"     CV={baseline_search.best_score_:.3f}, F1@prog={baseline_f1:.3f} "
            f"({baseline_src}), prog={baseline_threshold:.2f} [{mark}]",
            flush=True,
        )

    estimators = get_base_estimators(scale_pos_weight, num_cols_used, cat_cols_used)

    for name, estimator in estimators.items():
        if verbose:
            print(f"  -> {name}...", flush=True)
        pipe = _build_pipeline(preprocessor, estimator, name)
        search = _run_search(pipe, name, X_train_fit, y_train_fit, scale_pos_weight)
        searches[name] = search

        tuned, threshold, tuning_f1, src = _wrap_with_threshold(
            search.best_estimator_, X_train_fit, y_train_fit, X_val, y_val,
        )
        pipelines[name] = tuned
        best_params[name] = search.best_params_
        thresholds[name] = threshold
        oof_f1_scores[name] = tuning_f1
        threshold_sources[name] = src

        # XGBoost: retrenuj z early stopping na zbiorze walidacyjnym
        if name == "xgboost":
            try:
                best_pipe = search.best_estimator_
                steps_xgb = best_pipe.named_steps
                X_val_t = steps_xgb["preprocess"].transform(
                    steps_xgb["engineer"].transform(X_val))
                X_fit_t = steps_xgb["preprocess"].transform(
                    steps_xgb["engineer"].transform(X_train_fit))
                xgb_clf = steps_xgb["model"]
                xgb_clf.set_params(n_estimators=500, early_stopping_rounds=20)
                xgb_clf.fit(X_fit_t, y_train_fit,
                            eval_set=[(X_fit_t, y_train_fit), (X_val_t, y_val)],
                            verbose=False)
                # XGBoost 1.x / 2.x kompatybilnosc
                best_iter = (getattr(xgb_clf, "best_iteration_", None)
                             or getattr(xgb_clf, "best_iteration", None)
                             or "?")
                if hasattr(xgb_clf, "get_booster") and best_iter == "?":
                    try:
                        best_iter = xgb_clf.get_booster().best_iteration
                    except Exception:
                        pass
                if verbose:
                    print(f"     XGB early stop: {best_iter} iteracji", flush=True)
                tuned, threshold, tuning_f1, src = _wrap_with_threshold(
                    best_pipe, X_train_fit, y_train_fit, X_val, y_val,
                )
                pipelines[name] = tuned
                thresholds[name] = threshold
                oof_f1_scores[name] = tuning_f1
                threshold_sources[name] = src
            except Exception as e:
                if verbose:
                    print(f"     XGB early stop BLAD: {e}", flush=True)

        joblib.dump(tuned, MODELS_DIR / f"{name}.joblib")
        pd.DataFrame(search.cv_results_).to_csv(MODELS_DIR / f"cv_results_{name}.csv", index=False)
        if verbose:
            mark = "OK" if tuning_f1 >= F1_TARGET else "ponizej celu"
            print(
                f"     CV={search.best_score_:.3f}, F1@prog={tuning_f1:.3f} ({src}), "
                f"prog={threshold:.2f} [{mark}]",
                flush=True,
            )

    # Soft voting: LR + SVM + NN
    voting_members = ["logistic_regression", "svm", "neural_network"]
    if all(n in searches for n in voting_members):
        name = "voting"
        if verbose:
            print(f"  -> {name} (LR+SVM+NN soft avg)...", flush=True)
        member_pipes = [searches[m].best_estimator_ for m in voting_members]
        voting_pipe = SoftVotingEnsemble(member_pipes)
        tuned, threshold, tuning_f1, src = _wrap_with_threshold(
            voting_pipe, X_train_fit, y_train_fit, X_val, y_val,
        )
        cv_score = float(np.nanmean([searches[m].best_score_ for m in voting_members]))
        pipelines[name] = tuned
        best_params[name] = {}
        thresholds[name] = threshold
        oof_f1_scores[name] = tuning_f1
        threshold_sources[name] = src
        searches[name] = _FitOnceResult(voting_pipe, best_score=cv_score)
        joblib.dump(tuned, MODELS_DIR / f"{name}.joblib")
        pd.DataFrame(searches[name].cv_results_).to_csv(
            MODELS_DIR / f"cv_results_{name}.csv", index=False
        )
        if verbose:
            mark = "OK" if tuning_f1 >= F1_TARGET else "ponizej celu"
            print(
                f"     CV={cv_score:.3f}, F1@prog={tuning_f1:.3f} ({src}), "
                f"prog={threshold:.2f} [{mark}]",
                flush=True,
            )

    # Soft voting: LR + SVM (bez MLP — tylko dwa najlepiej generalizujace modele)
    voting_lr_svm_members = ["logistic_regression", "svm"]
    if all(n in searches for n in voting_lr_svm_members):
        name = "voting_lr_svm"
        if verbose:
            print(f"  -> {name} (LR+SVM soft avg)...", flush=True)
        member_pipes_2 = [searches[m].best_estimator_ for m in voting_lr_svm_members]
        voting_pipe_2 = SoftVotingEnsemble(member_pipes_2)
        tuned, threshold, tuning_f1, src = _wrap_with_threshold(
            voting_pipe_2, X_train_fit, y_train_fit, X_val, y_val,
        )
        cv_score_2 = float(np.nanmean([searches[m].best_score_ for m in voting_lr_svm_members]))
        pipelines[name] = tuned
        best_params[name] = {}
        thresholds[name] = threshold
        oof_f1_scores[name] = tuning_f1
        threshold_sources[name] = src
        searches[name] = _FitOnceResult(voting_pipe_2, best_score=cv_score_2)
        joblib.dump(tuned, MODELS_DIR / f"{name}.joblib")
        if verbose:
            mark = "OK" if tuning_f1 >= F1_TARGET else "ponizej celu"
            print(
                f"     CV={cv_score_2:.3f}, F1@prog={tuning_f1:.3f} ({src}), "
                f"prog={threshold:.2f} [{mark}]",
                flush=True,
            )

    # ── Warianty top-N cech (per-model, MAX_SELECTED_FEATURES_PER_MODEL) ────────
    top10_cols_per_model: dict[str, list[str]] = {}   # base_name → selected cols
    top10_names: list[str] = []

    if verbose:
        print(
            f"\n--- Warianty top-{MAX_SELECTED_FEATURES_PER_MODEL} cech per model ---",
            flush=True,
        )

    for base_name in list(estimators.keys()):
        if base_name not in pipelines:
            continue
        name = f"{base_name}_top10"
        if verbose:
            print(f"  -> {name}  [wyznaczam cechy...]", flush=True)

        # ── Wyznacz top-k cech charakterystycznych dla tego modelu ───────
        model_top_k = _get_top_k_cols_for_model(
            base_name,
            pipelines[base_name],
            num_cols_used,
            cat_cols_used,
            k=MAX_SELECTED_FEATURES_PER_MODEL,
            X_val=X_val,
            y_val=y_val,
        )
        top10_cols_per_model[base_name] = model_top_k

        model_top_num = [c for c in model_top_k if c in set(num_cols_used)]
        model_top_cat = [c for c in model_top_k if c in set(cat_cols_used)]
        model_preprocessor = build_preprocessor(model_top_num, model_top_cat)
        model_estimator = get_base_estimators(
            scale_pos_weight, model_top_num, model_top_cat
        )[base_name]

        # Przejdź hiperparametry z bazowego wyszukiwania
        if base_name in searches:
            base_params_top10 = {
                k[len("model__"):]: v
                for k, v in searches[base_name].best_params_.items()
                if k.startswith("model__")
            }
            if base_params_top10:
                try:
                    model_estimator.set_params(**base_params_top10)
                except Exception:
                    pass

        pipe = _build_pipeline_top10(
            model_preprocessor, model_estimator, base_name, model_top_k
        )
        pipe.fit(X_train_fit, y_train_fit)

        tuned, threshold, tuning_f1, src = _wrap_with_threshold(
            pipe, X_train_fit, y_train_fit, X_val, y_val,
        )
        pipelines[name] = tuned
        top10_names.append(name)
        best_params[name] = searches.get(base_name, _FitOnceResult(pipe)).best_params_
        thresholds[name] = threshold
        oof_f1_scores[name] = tuning_f1
        threshold_sources[name] = src

        joblib.dump(tuned, MODELS_DIR / f"{name}.joblib")
        if verbose:
            mark = "OK" if tuning_f1 >= F1_TARGET else "ponizej celu"
            print(
                f"     cechy={model_top_k[:4]}...  "
                f"F1@prog={tuning_f1:.3f} ({src}), prog={threshold:.2f} [{mark}]",
                flush=True,
            )

    # voting_top10: LR+SVM+NN — każdy ze swoimi cechami (union przez ensemble)
    voting_top10_members = [
        "logistic_regression_top10", "svm_top10", "neural_network_top10"
    ]
    if all(n in pipelines for n in voting_top10_members):
        name = "voting_top10"
        if verbose:
            print(
                f"  -> {name} "
                f"(LR+SVM+NN top{MAX_SELECTED_FEATURES_PER_MODEL} soft avg)...",
                flush=True,
            )
        member_pipes_top10 = [pipelines[m].pipeline for m in voting_top10_members]
        voting_pipe_top10 = SoftVotingEnsemble(member_pipes_top10)
        tuned, threshold, tuning_f1, src = _wrap_with_threshold(
            voting_pipe_top10, X_train_fit, y_train_fit, X_val, y_val,
        )
        cv_score = float(np.nanmean([
            searches.get(
                m.replace("_top10", ""), _FitOnceResult(voting_pipe_top10)
            ).best_score_
            for m in voting_top10_members
        ]))
        pipelines[name] = tuned
        top10_names.append(name)
        best_params[name] = {}
        thresholds[name] = threshold
        oof_f1_scores[name] = tuning_f1
        threshold_sources[name] = src
        searches[name] = _FitOnceResult(voting_pipe_top10, best_score=cv_score)
        joblib.dump(tuned, MODELS_DIR / f"{name}.joblib")
        if verbose:
            mark = "OK" if tuning_f1 >= F1_TARGET else "ponizej celu"
            print(
                f"     CV={cv_score:.3f}, F1@prog={tuning_f1:.3f} ({src}), "
                f"prog={threshold:.2f} [{mark}]",
                flush=True,
            )

    meta = {
        "num_cols": num_cols_used,
        "cat_cols": cat_cols_used,
        "all_num_cols": all_num_cols,
        "all_cat_cols": all_cat_cols,
        "target": "Attrition",
        "class_labels": {0: "Zostaje", 1: "Rezygnacja (Attrition)"},
        "cv_folds": CV_FOLDS,
        "cv_scoring": CV_SCORING,
        "f1_target": F1_TARGET,
        "preprocessing": f"wszystkie cechy, SMOTE ({SMOTE_STRATEGY}), Fbeta(beta=1.5), scaler (LR/MLP/SVM/NN)",
        "threshold_tuning": f"Fbeta(beta=1.5)>={F1_TARGET}: oof (bez train/smote_train)",
        "val_size": VAL_SIZE,
        "n_train_fit": len(X_train_fit),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "best_params": best_params,
        "best_cv_scores": {name: searches[name].best_score_ for name in searches},
        "tuning_f1_at_threshold": oof_f1_scores,
        "threshold_sources": threshold_sources,
        "decision_thresholds": thresholds,
        "top10_cols_per_model": top10_cols_per_model,
        "top10_model_names": top10_names,
        "max_selected_features": MAX_SELECTED_FEATURES_PER_MODEL,
    }
    joblib.dump(meta, MODELS_DIR / "metadata.joblib")

    return {
        "pipelines": pipelines,
        "searches": searches,
        "X_train": X_train_fit,
        "X_test": X_test,
        "X_val": X_val,
        "y_train": y_train_fit,
        "y_test": y_test,
        "y_val": y_val,
        "preprocessor_fitted": pipelines["xgboost"].pipeline.named_steps["preprocess"],
        "num_cols": num_cols_used,
        "cat_cols": cat_cols_used,
        "all_num_cols": all_num_cols,
        "all_cat_cols": all_cat_cols,
        "best_params": best_params,
        "thresholds": thresholds,
        "oof_f1_scores": oof_f1_scores,
        "threshold_sources": threshold_sources,
        "top10_cols_per_model": top10_cols_per_model,
        "top10_names": top10_names,
    }
