"""Prog decyzyjny optymalizowany pod F1 (docelowo >= F1_TARGET)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, fbeta_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import CV_FOLDS, F1_TARGET, RANDOM_STATE, THRESHOLD_BETA, VAL_SIZE


def find_best_threshold(
    y_true,
    y_proba_positive: np.ndarray,
    step: float | None = None,
    beta: float = 1.0,
) -> tuple[float, float]:
    """Prog maksymalizujacy Fbeta (beta=1 → F1; beta>1 → faworyzuje recall)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba_positive, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    def _score(y_true, y_pred):
        if beta == 1.0:
            return f1_score(y_true, y_pred, zero_division=0)
        return fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    # Krzywa precision-recall — wszystkie sensowne progi
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    fb_scores = np.zeros_like(thresholds, dtype=float)
    for i, thr in enumerate(thresholds):
        fb_scores[i] = _score(y_true, (proba >= thr).astype(int))

    if len(fb_scores):
        best_idx = int(np.argmax(fb_scores))
        return float(thresholds[best_idx]), float(fb_scores[best_idx])

    if step is None:
        step = 0.005
    best_threshold = 0.5
    best_fb = 0.0
    for threshold in np.arange(0.01, 0.995, step):
        y_pred = (proba >= threshold).astype(int)
        sc = _score(y_true, y_pred)
        if sc > best_fb:
            best_fb = sc
            best_threshold = float(threshold)
    return best_threshold, best_fb


def find_threshold_for_target(
    y_true,
    y_proba_positive: np.ndarray,
    target_f1: float = F1_TARGET,
) -> tuple[float, float]:
    """Najnizszy prog, przy ktorym F1 >= target (jesli istnieje); inaczej max F1."""
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba_positive, dtype=float)

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    best_at_target: tuple[float, float] | None = None
    best_any = (0.5, 0.0)

    for thr in thresholds:
        f1 = f1_score(y_true, (proba >= thr).astype(int), zero_division=0)
        if f1 > best_any[1]:
            best_any = (float(thr), float(f1))
        if f1 >= target_f1:
            if best_at_target is None or thr < best_at_target[0]:
                best_at_target = (float(thr), float(f1))

    if best_at_target is not None:
        return best_at_target
    return best_any


def _resampled_train_proba(pipeline, X_train, y_train) -> tuple[np.ndarray, np.ndarray]:
    """P(y=1) na zbiorze po SMOTE (zblizonym do danych treningowych pipeline)."""
    steps = pipeline.named_steps
    X_prep = steps["preprocess"].transform(X_train)
    X_res, y_res = steps["smote"].fit_resample(X_prep, y_train)
    if "scaler" in steps:
        X_res = steps["scaler"].transform(X_res)
    proba = steps["model"].predict_proba(X_res)[:, 1]
    return np.asarray(y_res), proba


def select_threshold(
    pipeline,
    X_train,
    y_train,
    f1_target: float = F1_TARGET,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    X_val=None,
    y_val=None,
    beta: float = THRESHOLD_BETA,
) -> tuple[float, float, str]:
    """
    Próg dobierany na zbiorze walidacyjnym (jeśli podany) lub OOF.
    Val to prawdziwe dane bez SMOTE — threshold transferuje lepiej na test.
    """
    if X_val is not None and y_val is not None:
        proba = pipeline.predict_proba(X_val)[:, 1]
        thr, fb = find_best_threshold(y_val, proba, beta=beta)
        return thr, fb, "val"

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    thr, fb = find_best_threshold(y_train, oof_proba, beta=beta)
    return thr, fb, "oof"


def find_threshold_from_oof(
    pipeline,
    X,
    y,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Zachowana kompatybilnosc: prog z pelnej strategii select_threshold."""
    thr, f1, _ = select_threshold(pipeline, X, y, F1_TARGET, cv_folds, random_state)
    return thr, f1


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Pipeline sklearn + dopasowany prog P(Attrition=1)."""

    def __init__(self, pipeline, threshold: float = 0.5, threshold_source: str = "oof"):
        self.pipeline = pipeline
        self.threshold = threshold
        self.threshold_source = threshold_source

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
