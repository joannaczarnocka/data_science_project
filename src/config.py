"""Konfiguracja projektu HR Attrition."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_FILENAME = "HR.csv"
PROCESSED_ENGINEERED = "hr_engineered.csv"
# Imputacja + OneHotEncoding + StandardScaler (bez SMOTE)
PROCESSED_PREPROCESSED = "hr_preprocessed.csv"
PREPROCESSING_PIPELINE_FILE = "preprocessing_pipeline.joblib"
TARGET = "Attrition"

# Kolumny do usuniecia (ID, stale, target)
DROP_COLS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]

CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Wyliczane w engineer_features() — nie pokazuj w formularzu Streamlit
ENGINEERED_NUM_COLS = [
    "IncomePerYearExp",
    "CompanyTenureRatio",
    "RoleStabilityRatio",
    "PromotionIntensity",
    "ManagerTenureRatio",
    "CompaniesPerYear",
    "HighOvertime",
    "AvgSatisfaction",
    "LowSatisfaction",
    "Stagnation",
    "YoungHighMobility",
    "IncomeVsDeptMedian",
    "CareerStageRatio",
    "OvertimeLowSat",
    "LongTimeNoPromotion",
    "OvertimeHighRiskRole",
    "IncomePerJobLevel",
    "TenureWithoutPromotion",
    "TravelBurden",
    # v3
    "IncomePerYearAtCompany",
    "OvertimeAndLowSatisfaction",
    "YoungAndOvertime",
    "IncomeVsJobLevel",
    "SatisfactionVariance",
    "TotalExperienceGap",
    "ManagerInstability",
    "FrequentJobSwitcher",
    "BurnoutRiskScore",
    "IncomePerDependent",
    "LateCareerNoPromotion",
    "CommuteRisk",
    "ExperienceMismatch",
    "StabilityComposite",
    "TravelFatigue",
    "LowIncomeHighTenure",
    "CareerGrowthIndex",
    # v4
    "LowJobInvolvement",
    "NoStockOptions",
    "StockRetentionRisk",
    "LowInvolvementOvertime",
    "NoTrainingLastYear",
    "TrainingDeficit",
    "HighPerfNoPromotion",
    "SalaryHikeVsPerformance",
    "TotalDissatisfaction",
    "InvolvementXSatisfaction",
    "RecentHireOvertime",
    "StockDissatisfactionRisk",
]

RANDOM_STATE = 42
TEST_SIZE = 0.4
VAL_SIZE = 0.2   # frakcja X_train → zbiór walidacyjny do doboru progu
CV_FOLDS = 5
CV_SCORING = "roc_auc"
F1_TARGET = 0.7
SMOTE_STRATEGY = 1.0
RANDOM_SEARCH_ITER = 50

THRESHOLD_BETA = 1.0
MAX_SELECTED_FEATURES_PER_MODEL = 20
