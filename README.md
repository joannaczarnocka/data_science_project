# HR Attrition — projekt Data Science

Predykcja rezygnacji pracownikow na podstawie `data/raw/HR.csv`.

## Struktura

```
PROJECT/
├── data/raw/HR.csv
├── notebooks/          # EDA, preprocessing, modeling
├── src/                # pipeline Python
├── scripts/train.py    # trenowanie modeli
├── app/streamlit_app.py
├── models/             # zapisane pipeline'y (.joblib)
└── reports/            # wykresy i CSV z metrykami
```

## Szybki start

**Instalacja (Windows / Linux / macOS):**

```bash
python scripts/setup_env.py
python scripts/run_train.py
python scripts/run_streamlit.py
```

`setup_env.py` tworzy `venv` z `--system-site-packages` i uzywa `trusted-host` dla pip (patrz `pip.ini`).

Po instalacji mozesz aktywowac venv:

- Windows: `venv\Scripts\activate`
- Linux/macOS: `source venv/bin/activate`

**Jesli `pip install` w venv pada (SSL):** uzyj globalnego Pythona:

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
python scripts/train.py
```

Jupyter (opcjonalnie): `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-notebooks.txt`

## Modele

| Model | Opis |
|-------|------|
| Logistic Regression | `class_weight=balanced`, GridSearchCV |
| Random Forest | feature importance, GridSearchCV |
| Neural Network | MLP, early stopping, GridSearchCV |
| XGBoost | gradient boosting, `scale_pos_weight`, GridSearchCV |
| SVM | `SVC` (RBF/linear), `StandardScaler`, GridSearchCV |
| MLP (sklearn) | `MLPClassifier` — sieć w scikit-learn |

**Preprocessing:** imputacja + one-hot → **SMOTE** → **StandardScaler** (LR/MLP/NN) → model (wszystkie cechy po feature engineering).

**Regularizacja:** L2 (LR, stacking), `alpha` (MLP), `C` (SVM), `reg_alpha`/`reg_lambda` (XGBoost), ograniczenie głębokości i `min_samples_leaf` (RF).

**Walidacja:** `StratifiedKFold` **5-fold**, `GridSearchCV` / `RandomizedSearchCV` (scoring=`f1`).

**Prog decyzyjny:** dopasowany pod F1 na predykcjach OOF (out-of-fold).

**Realistyczny F1 na test:** ~0.52–0.58 (niezbalansowany Attrition ~16%). Cel F1>0.8 nie jest osiagalny na uczciwym holdout bez wiecej danych / innej definicji problemu — zob. komunikat po `train.py`.
Wyniki CV: `models/cv_results_*.csv`, `reports/cv_summary.csv`.

## Feature engineering

**Baseline (v1):**
- `IncomePerYearExp`, `CompanyTenureRatio`, `RoleStabilityRatio`
- `PromotionIntensity`, `ManagerTenureRatio`, `CompaniesPerYear`
- `HighOvertime`, `OvertimeLowSat`, `LowSatisfaction`, `AvgSatisfaction`
- `Stagnation`, `YoungHighMobility`, `LongTimeNoPromotion`
- `IncomeVsDeptMedian`, `CareerStageRatio`
- Poprawka `0n-Travel` → `Non-Travel`

**Rozszerzenie (v2):**
| Cecha | Formuła | Interpretacja |
|-------|---------|---------------|
| `OvertimeHighRiskRole` | `OverTime AND JobRole ∈ {Sales Rep, Lab Tech}` | Nadgodziny w rolach o wysokiej rotacji |
| `IncomePerJobLevel` | `MonthlyIncome / JobLevel` | Sygnał niedopłacenia względem poziomu stanowiska |
| `TenureWithoutPromotion` | `YearsAtCompany − YearsSinceLastPromotion` | Lata stażu do ostatniego awansu |
| `TravelBurden` | `DistanceFromHome × travel_weight` | Łączny ciężar podróżowania |

## Feature importance

- LR: wartosc bezwzgledna wspolczynnikow
- RF: `feature_importances_`
- MLP: permutation importance

Raporty w `reports/feature_importance_*.png` i `.csv`.

## Najlepszy model

Po treningu wybierany jest model z najwyzszym **F1 na zbiorze testowym** (holdout 20%) i zapisywany jako:

- `models/best_model.joblib` — pipeline uzywany w Streamlit
- `reports/best_model.txt` — opis metryk i progu decyzyjnego

Aktualizacja bez pelnego treningu: `python scripts/update_best_model.py`

## Aplikacja Streamlit

Formularz boczny → **prognoza z najlepszego modelu** (`best_model.joblib`); opcjonalnie porownanie wszystkich modeli.
