"""Czyszczenie danych i inzynieria cech (baseline)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_COLS,
    DROP_COLS,
    ENGINEERED_NUM_COLS,
    MODELS_DIR,
    PREPROCESSING_PIPELINE_FILE,
    PROCESSED_PREPROCESSED,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Wybiera podzbior kolumn z inżynierowanego DataFrame po kroku engineer."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return df[self.columns]


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible krok 0 pipeline: surowy DataFrame HR → inżynierowane cechy.

    Dzięki temu zapisany .joblib jest samowystarczalny — można podać surowe dane.
    engineer_features jest idempotentna: aplikacja na już-inżynierowanych danych
    daje ten sam wynik (pola są nadpisywane tymi samymi wartościami).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return engineer_features(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Poprawki jakosci danych."""
    out = df.copy()
    out["BusinessTravel"] = out["BusinessTravel"].replace({"0n-Travel": "Non-Travel"})
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline feature engineering."""
    out = clean_data(df)
    twy = out["TotalWorkingYears"] + 1
    yac = out["YearsAtCompany"] + 1

    out["IncomePerYearExp"] = out["MonthlyIncome"] / twy
    out["CompanyTenureRatio"] = out["YearsAtCompany"] / twy
    out["RoleStabilityRatio"] = out["YearsInCurrentRole"] / yac
    out["PromotionIntensity"] = out["YearsSinceLastPromotion"] / yac
    out["ManagerTenureRatio"] = out["YearsWithCurrManager"] / yac
    out["CompaniesPerYear"] = out["NumCompaniesWorked"] / twy
    out["HighOvertime"] = _is_overtime_yes(out["OverTime"]).astype(int)

    sat_cols = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    out["AvgSatisfaction"] = out[sat_cols].mean(axis=1)
    out["LowSatisfaction"] = (out["AvgSatisfaction"] <= 2).astype(int)
    out["Stagnation"] = (
        (out["YearsSinceLastPromotion"] >= 4) & (out["YearsAtCompany"] >= 3)
    ).astype(int)
    out["YoungHighMobility"] = (
        (out["Age"] < 35) & (out["NumCompaniesWorked"] >= 3)
    ).astype(int)

    dept_median_income = out.groupby("Department")["MonthlyIncome"].transform("median")
    out["IncomeVsDeptMedian"] = out["MonthlyIncome"] / (dept_median_income + 1)
    out["CareerStageRatio"] = out["TotalWorkingYears"] / (out["Age"] - 18).clip(lower=1)
    out["OvertimeLowSat"] = (
        _is_overtime_yes(out["OverTime"]) & (out["AvgSatisfaction"] <= 2.5)
    ).astype(int)
    out["LongTimeNoPromotion"] = (
        out["YearsSinceLastPromotion"] >= 3
    ).astype(int)

    # OverTime × JobRole — nadgodziny w rolach o wysokiej rotacji
    _high_risk_roles = {"Sales Representative", "Laboratory Technician"}
    out["OvertimeHighRiskRole"] = (
        _is_overtime_yes(out["OverTime"]) &
        out["JobRole"].astype(str).isin(_high_risk_roles)
    ).astype(int)

    # MonthlyIncome / JobLevel — sygnał niedopłacenia względem poziomu stanowiska
    out["IncomePerJobLevel"] = out["MonthlyIncome"] / out["JobLevel"].clip(lower=1)

    # YearsAtCompany - YearsSinceLastPromotion — ile lat minęło od awansu do dziś
    # Mała wartość = awans tuż przed odejściem lub brak awansu przez cały staż
    out["TenureWithoutPromotion"] = (
        out["YearsAtCompany"] - out["YearsSinceLastPromotion"]
    ).clip(lower=0)

    # DistanceFromHome × BusinessTravel — łączny ciężar podróżowania
    _travel_weight = out["BusinessTravel"].astype(str).map(
        {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    ).fillna(1)
    out["TravelBurden"] = out["DistanceFromHome"] * _travel_weight

    # v3 features
    _overtime_bin = out["HighOvertime"]  # 0/1 already computed above
    _travel_enc = _travel_weight.astype(int)

    out["IncomePerYearAtCompany"] = out["MonthlyIncome"] / (out["YearsAtCompany"] + 1)
    out["OvertimeAndLowSatisfaction"] = (
        _overtime_bin.astype(bool) & (out["JobSatisfaction"] <= 2)
    ).astype(int)
    out["YoungAndOvertime"] = (
        (out["Age"] < 30) & _overtime_bin.astype(bool)
    ).astype(int)
    out["IncomeVsJobLevel"] = out["MonthlyIncome"] / (out["JobLevel"] + 1)
    out["SatisfactionVariance"] = out[sat_cols].std(axis=1)
    out["TotalExperienceGap"] = (out["TotalWorkingYears"] - out["YearsAtCompany"]).clip(lower=0)
    out["ManagerInstability"] = (out["YearsAtCompany"] - out["YearsWithCurrManager"]).clip(lower=0)
    out["FrequentJobSwitcher"] = (out["NumCompaniesWorked"] >= 5).astype(int)
    out["BurnoutRiskScore"] = (
        2 * _overtime_bin + (4 - out["WorkLifeBalance"]) + (4 - out["JobSatisfaction"])
    )
    out["IncomePerDependent"] = out["MonthlyIncome"] / (out["NumCompaniesWorked"] + 1)
    out["LateCareerNoPromotion"] = (
        (out["Age"] > 40) & (out["YearsSinceLastPromotion"] > 5)
    ).astype(int)
    out["CommuteRisk"] = out["DistanceFromHome"] * _overtime_bin
    out["ExperienceMismatch"] = out["TotalWorkingYears"] / (out["JobLevel"] + 1)
    out["StabilityComposite"] = (
        out["YearsWithCurrManager"] + out["YearsInCurrentRole"] + out["YearsAtCompany"]
    )
    out["TravelFatigue"] = _travel_enc * out["DistanceFromHome"] * _overtime_bin
    out["LowIncomeHighTenure"] = out["YearsAtCompany"] / (out["MonthlyIncome"] + 1)
    out["CareerGrowthIndex"] = out["JobLevel"] / (out["TotalWorkingYears"] + 1)

    # v4: JobInvolvement, StockOptions, Training, Performance × Promotion
    out["LowJobInvolvement"] = (out["JobInvolvement"] <= 2).astype(int)
    out["NoStockOptions"] = (out["StockOptionLevel"] == 0).astype(int)
    # Brak akcji + nadgodziny = wysokie ryzyko odejścia
    out["StockRetentionRisk"] = (4 - out["StockOptionLevel"]) * _overtime_bin
    # Niskie zaangażowanie + nadgodziny = wypalenie bez motywacji
    out["LowInvolvementOvertime"] = (
        (out["JobInvolvement"] <= 2) & _overtime_bin.astype(bool)
    ).astype(int)
    # Brak szkoleń = niedoinwestowany pracownik
    out["NoTrainingLastYear"] = (out["TrainingTimesLastYear"] == 0).astype(int)
    out["TrainingDeficit"] = (out["TrainingTimesLastYear"] <= 1).astype(int)
    # Wysoka ocena + brak awansu = frustracja
    out["HighPerfNoPromotion"] = (
        (out["PerformanceRating"] >= 3) & (out["YearsSinceLastPromotion"] >= 4)
    ).astype(int)
    # Podwyżka niska względem oceny wydajności
    out["SalaryHikeVsPerformance"] = out["PercentSalaryHike"] / out["PerformanceRating"].clip(lower=1)
    # Suma niezadowolenia we wszystkich wymiarach
    out["TotalDissatisfaction"] = (
        (4 - out["JobSatisfaction"]) +
        (4 - out["WorkLifeBalance"]) +
        (4 - out["EnvironmentSatisfaction"]) +
        (4 - out["RelationshipSatisfaction"])
    )
    # Zaangażowanie × satysfakcja — niska na obu = silny predyktor odejścia
    out["InvolvementXSatisfaction"] = out["JobInvolvement"] * out["AvgSatisfaction"]
    # Nowy pracownik + nadgodziny = duże ryzyko natychmiastowego odejścia
    out["RecentHireOvertime"] = (
        (out["YearsAtCompany"] <= 2) & _overtime_bin.astype(bool)
    ).astype(int)
    # Wysoki stock risk × ogólna niezadowolenie
    out["StockDissatisfactionRisk"] = (4 - out["StockOptionLevel"]) * out["TotalDissatisfaction"]

    return out


def _is_overtime_yes(series: pd.Series) -> pd.Series:
    """OverTime jako Yes/No lub 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int) == 1
    return series.astype(str).str.strip().isin({"Yes", "1", "True", "yes"})


def _is_textual_column(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Zwraca (numeryczne, kategoryczne) kolumny predykcyjne."""
    exclude = set(DROP_COLS + [TARGET])
    feature_cols = [c for c in df.columns if c not in exclude]

    config_cat = [c for c in CATEGORICAL_COLS if c in feature_cols]
    text_cat = [c for c in feature_cols if c not in config_cat and _is_textual_column(df[c])]
    cat_cols = list(dict.fromkeys(config_cat + text_cat))
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return num_cols, cat_cols


def cast_categorical_columns(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Tekstowe kolumny kategoryczne jako str (wymagane przez OneHotEncoder)."""
    out = df.copy()
    for col in cat_cols:
        if col not in out.columns:
            continue
        out[col] = out[col].astype("string").fillna("Unknown").astype(str)
    return out


def build_categorical_encoder() -> OneHotEncoder:
    """One-hot dla zmiennych kategorycznych (tekstowych i dyskretnych)."""
    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Preprocessor: imputacja + OneHotEncoding kolumn kategorycznych."""
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_categorical_encoder()),
        ]
    )
    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")



def build_preprocessing_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Pelny preprocessing do zapisu: imputacja + OneHotEncoding, potem StandardScaler."""
    return Pipeline(
        [
            ("encode", build_preprocessor(num_cols, cat_cols)),
            ("scale", StandardScaler()),
        ]
    )


def get_input_num_cols(num_cols: list[str]) -> list[str]:
    """Kolumny numeryczne do formularza (bez wyliczonych)."""
    return [c for c in num_cols if c not in ENGINEERED_NUM_COLS]


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Przygotuj X, y oraz listy kolumn."""
    engineered = engineer_features(df)
    num_cols, cat_cols = get_feature_columns(engineered)
    X = cast_categorical_columns(engineered[num_cols + cat_cols], cat_cols)
    y = engineered[TARGET]
    return X, y, num_cols, cat_cols


def transform_full_preprocessing(
    prep_pipeline: Pipeline,
    X: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Pelny preprocessing: imputacja, OneHotEncoding, StandardScaler.
    Obejmuje tez kolumny z feature engineering (sa w num_cols).
    """
    encoder = prep_pipeline.named_steps["encode"]
    names = get_feature_names(encoder, num_cols, cat_cols)
    values = prep_pipeline.transform(X)[0]
    return values, names


def fit_and_save_preprocessing_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Dopasuj i zapisz pipeline preprocessingu (jak hr_preprocessed.csv)."""
    pipeline = build_preprocessing_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)
    path = MODELS_DIR / PREPROCESSING_PIPELINE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return pipeline


def get_feature_names(preprocessor: ColumnTransformer, num_cols: list[str], cat_cols: list[str]) -> list[str]:
    """Nazwy cech po transformacji (numeryczne + kolumny po OneHotEncoding)."""
    names = list(num_cols)
    if cat_cols and "cat" in preprocessor.named_transformers_:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        names.extend(encoder.get_feature_names_out(cat_cols))
    return names


def _fit_transform_preprocessing(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit_on_train: bool,
) -> np.ndarray:
    if fit_on_train:
        from sklearn.model_selection import train_test_split

        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        pipeline.fit(X_train, y_train)
        return pipeline.transform(X)
    return pipeline.fit_transform(X, y)


def build_preprocessed_dataframe(
    df: pd.DataFrame | None = None,
    *,
    fit_on_train: bool = True,
    scale: bool = True,
) -> pd.DataFrame:
    """
    Zbiór po pelnym preprocessingu (bez SMOTE).

    Kolejnosc: imputacja -> OneHotEncoding -> StandardScaler (jesli scale=True).
    Dopasowanie na zbiorze treningowym (80%), transformacja wszystkich wierszy.
    """
    if df is None:
        from data_loader import load_hr

        df = load_hr()

    X, y, num_cols, cat_cols = get_X_y(df)

    if scale:
        pipeline = build_preprocessing_pipeline(num_cols, cat_cols)
        X_transformed = _fit_transform_preprocessing(pipeline, X, y, fit_on_train=fit_on_train)
        encoder = pipeline.named_steps["encode"]
    else:
        encoder = build_preprocessor(num_cols, cat_cols)
        if fit_on_train:
            from sklearn.model_selection import train_test_split

            X_train, _, y_train, _ = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            encoder.fit(X_train, y_train)
            X_transformed = encoder.transform(X)
        else:
            X_transformed = encoder.fit_transform(X, y)

    feature_names = get_feature_names(encoder, num_cols, cat_cols)
    out = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    out[TARGET] = y.values
    return out


def save_preprocessed_dataset(
    df: pd.DataFrame | None = None,
    filename: str = PROCESSED_PREPROCESSED,
) -> Path:
    """Zapisz zbiór po preprocessingu (OHE + standaryzacja) do data/processed/."""
    from data_loader import save_processed

    preprocessed = build_preprocessed_dataframe(df, scale=True)
    return save_processed(preprocessed, filename)
