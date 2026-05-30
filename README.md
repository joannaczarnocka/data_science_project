# HR Attrition — projekt Data Science

Predykcja rezygnacji pracownikow (`Attrition`) na podstawie zbioru **IBM HR Analytics** (1470 pracownikow, ~16% klasy pozytywnej).

**Autor:** Joanna Czarnocka

## Wynik

| Metryka | Wartosc |
|---------|---------|
| **Najlepszy model** | **Voting LR+SVM** (soft voting) |
| **F1 (klasa Rezygnuje)** | **0.598** |
| **ROC-AUC** | **0.821** |
| **Precision / Recall** | 0.560 / 0.642 |
| **Overfit (ΔF1 train-test)** | **0.061** |
| Macierz pomylek (test, 588 prob.) | TN=450, FP=43, FN=33, TP=62 |

Realistyczny pulap F1 dla syntetycznego IBM HR Attrition (~16% klasy) to **0.56–0.65**. Wyzej wymaga wiekszego zbioru lub innego podejscia.

## Struktura projektu

```
PROJECT/
├── data/raw/HR.csv               # surowy zbior danych
├── data/processed/               # po preprocessingu
├── notebooks/                    # 01_eda, 02_preprocessing, 03_modeling, 04_clustering
├── src/
│   ├── config.py                 # stale: TEST_SIZE, SMOTE_STRATEGY, MAX_SELECTED_FEATURES_PER_MODEL
│   ├── data_loader.py            # ladowanie HR.csv
│   ├── features.py               # FeatureEngineeringTransformer, ColumnSelector, get_X_y, OHE
│   ├── model.py                  # train_all_models, definicje 10 modeli + warianty top-20
│   ├── tf_model.py               # TFNeuralNetworkClassifier (Keras 3, sklearn-compatible)
│   ├── threshold.py              # OOF/val dobor progu, ThresholdClassifier
│   ├── evaluate.py               # metryki, feature importance
│   └── best_model.py             # wybor i zapis best_model.joblib
├── scripts/
│   ├── train.py                  # pelny trening wszystkich modeli (30-90 min)
│   ├── save_model_reports.py     # raport per-model i per-klasa
│   ├── cluster_employees.py      # KMeans k=4, segmentacja pracownikow
│   ├── run_streamlit.py          # uruchomienie aplikacji webowej
│   └── _show_*.py                # narzedzia do raportowania metryk
├── app/streamlit_app.py          # interaktywna aplikacja predykcyjna
├── models/                       # zapisane pipeline'y (.joblib) — 21 wariantow
├── reports/                      # wykresy PNG, CSV z metrykami, feature importance
└── presentation/                 # HR_Attrition_Projekt.pptx + .pdf (15 slajdow)
```

## Szybki start

```bash
# 1. Instalacja zaleznosci (Windows / Linux / macOS)
python scripts/setup_env.py

# 2. Trening modeli (30-90 min)
python scripts/run_train.py

# 3. Aplikacja interaktywna
python scripts/run_streamlit.py
```

Aktywacja venv recznie:
- Windows: `venv\Scripts\activate`
- Linux/macOS: `source venv/bin/activate`

Jesli `pip install` w venv pada (SSL):
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
python scripts/train.py
```

Jupyter (opcjonalnie): `pip install -r requirements-notebooks.txt`

## Modele

### Bazowe (8)

| Model | Balansowanie klas | Regularyzacja | Hiperparametry |
|-------|------------------|---------------|----------------|
| **Logistic Regression** | SMOTE (1.0) | L1/L2/ElasticNet, C=0.001 | GridSearch po `penalty` × `C` |
| **SVM** | `class_weight=balanced` | C ∈ [0.001, 0.1] | GridSearch po `C` × `kernel` |
| **Neural Network (MLP)** | `class_weight` (sklearn) | alpha=[0.05, 0.5] | RandomizedSearch (32, 64 jednostek) |
| **TF Neural Network** | `class_weight` w `fit()` | Dropout 0.3, L2=0.01, BatchNorm | Dense(64→32→16) + early stopping |
| **Random Forest** | SMOTE (1.0) | max_depth ≤ 6, min_samples_leaf ≥ 15 | RandomizedSearch |
| **Balanced Random Forest** | sampling_strategy='all' + SMOTE | max_depth ≤ 6 | RandomizedSearch |
| **XGBoost** | scale_pos_weight × 1.5 | reg_alpha=2-10, reg_lambda=5-20, depth=1-3 | **Early stopping (val)** |
| **CatBoost** | auto_class_weights='Balanced' | l2_leaf_reg=15-50, depth=3-5 | RandomizedSearch |

### Ensemble (3)

- **Voting LR+SVM** ← **najlepszy model** (soft voting, P_avg = (P_LR + P_SVM) / 2)
- **Voting LR+SVM+MLP** (3-elementowe soft voting)
- **Stacking LR+SVM → meta-LR** (5-fold CV → LogReg jako meta-klasyfikator)

### Warianty top-20 cech per model (9)

Kazdy model bazowy ma drugi wariant `{nazwa}_top10` (nazwa historyczna — faktycznie 20 cech, `MAX_SELECTED_FEATURES_PER_MODEL=20`) gdzie waznosc cech wyznaczana jest metoda dobrana do klasy modelu:

| Klasa modelu | Metoda waznosci |
|--------------|----------------|
| LR | `|coef_|` (agregacja OHE → kolumna oryginalna) |
| RF, BRF, XGB | `feature_importances_` (Gini / gain) |
| CatBoost | natywna `feature_importances_` |
| SVM, MLP, TF, Stacking | **permutation importance** (5 powtorzen, F1) |

**Lacznie: 21 modeli** zapisanych jako `.joblib`.

## Pipeline

```
surowe dane (DataFrame)
   ↓
[engineer]    FeatureEngineeringTransformer  (88 cech inzynierowanych — v1/v2/v3/v4)
   ↓
[select]      (tylko warianty top-20: ColumnSelector wybiera top-N cech wg waznosci)
   ↓
[preprocess]  imputacja median/mod + OneHotEncoder
   ↓
[smote]       SMOTE 1.0 (tylko LR, RF, BRF) — SVM/MLP/TF uzywaja class_weight
   ↓
[scaler]      StandardScaler (LR/SVM/MLP/TF)
   ↓
[model]       klasyfikator
```

Wszystkie pipeline'y maja `engineer` jako step 0 — **mozesz podac surowe dane**, pipeline sam zrobi feature engineering. `engineer_features` jest idempotentne: wywolanie na juz-zaengineerowanych danych nadpisuje cechy pochodne tymi samymi wartosciami.

## Feature engineering — 88 cech inzynierowanych

### v1/v2 (19 cech) — Stabilnosc i kariera
`IncomePerYearExp`, `CompanyTenureRatio`, `RoleStabilityRatio`, `PromotionIntensity`, `ManagerTenureRatio`, `CompaniesPerYear`, `HighOvertime`, `AvgSatisfaction`, `LowSatisfaction`, `Stagnation`, `YoungHighMobility`, `IncomeVsDeptMedian`, `CareerStageRatio`, `OvertimeLowSat`, `LongTimeNoPromotion`, `OvertimeHighRiskRole`, `IncomePerJobLevel`, `TenureWithoutPromotion`, `TravelBurden`

### v3 (17 cech) — Ryzyko i wypalenie
`IncomePerYearAtCompany`, `OvertimeAndLowSatisfaction`, `YoungAndOvertime`, `IncomeVsJobLevel`, `SatisfactionVariance`, `TotalExperienceGap`, `ManagerInstability`, `FrequentJobSwitcher`, **`BurnoutRiskScore`**, `IncomePerDependent`, `LateCareerNoPromotion`, `CommuteRisk`, `ExperienceMismatch`, **`StabilityComposite`**, `TravelFatigue`, `LowIncomeHighTenure`, `CareerGrowthIndex`

### v4 (12 cech) — Zaangazowanie i opcje akcji
`LowJobInvolvement`, `NoStockOptions`, **`StockRetentionRisk`**, `LowInvolvementOvertime`, `NoTrainingLastYear`, `TrainingDeficit`, `HighPerfNoPromotion`, `SalaryHikeVsPerformance`, `TotalDissatisfaction`, **`InvolvementXSatisfaction`**, `RecentHireOvertime`, **`StockDissatisfactionRisk`**

**Pelne formuly:** notebook `notebooks/02_preprocessing.ipynb` lub `src/features.py:engineer_features()`.

## SMOTE vs class_weight — uzasadnienie

Eksperymenty wykazaly, ze SMOTE 1:1 powoduje overfitting przy modelach o duzej pojemnosci (SVM, MLP, TF). Zastapienie SMOTE przez `class_weight` zmniejszylo `ΔF1 train-test` z 0.19 → 0.06 dla SVM (kluczowa zmiana — SVM stal sie konkurencyjnym modelem). Strategie per model:

| Model | Metoda balansowania | Powod |
|-------|--------------------|----- |
| LR, RF, BRF | SMOTE strategy=1.0 | Wymaga pelnego balansu dla konwergencji |
| **SVM, MLP, TF** | **class_weight** | SMOTE powodowal nadmierne dopasowanie |
| XGBoost | `scale_pos_weight=5.2` | Natywne wazenie |
| CatBoost | `auto_class_weights='Balanced'` | Natywne wazenie |

## Walidacja i dobor progu

- **Podzial danych:** train_fit 48% / val 12% / test 40% (stratify)
- **Cross-validation:** StratifiedKFold 5-fold, scoring=`roc_auc`
- **Wyszukiwanie hiperparametrow:** GridSearchCV (LR, SVM) / RandomizedSearchCV 50 iteracji (drzewa, NN)
- **Prog decyzyjny:** dobierany na zbiorze **val** (177 probek) — uczciwe, bez wycieku z test
- **OOF fallback:** jesli val niedostepny, prog z `cross_val_predict` (5-fold)

## Feature importance

Wykresy i CSV w `reports/feature_importance_*` dla kazdego modelu. Heatmapa porownawcza top-15 cech: `reports/feature_importance_comparison_heatmap.png`.

**Kluczowe predyktory we wszystkich modelach:**
- `OverTime` / `OvertimeLowSat` — nadgodziny
- `StockDissatisfactionRisk` / `NoStockOptions` — brak opcji akcji
- `InvolvementXSatisfaction` — iloczyn zaangazowania i satysfakcji
- `CompaniesPerYear` / `FrequentJobSwitcher` — historia zmian pracodawcow
- `Age` + `MonthlyIncome` — mlodzi i nisko oplacani

## Clustering pracownikow

KMeans k=4 na 21 cechach (`scripts/cluster_employees.py`) — 4 segmenty:

| Klaster | Nazwa | N | Attrition | Priorytet HR |
|---------|-------|---|-----------|---------------|
| 0 | Nomadzi | 447 (30%) | 19% | Sciezka kariery, mentoring |
| 1 | **Starterzy** | 431 (29%) | **22% (najwyzszy)** | Podwyzki entry-level, buddy |
| 2 | Lojalni Stagnujacy | 407 (28%) | 10% | Lateral moves, projekty |
| 3 | Seniorzy | 185 (13%) | 9% (najnizszy) | Mentoring odwrocony, elastycznosc |

Raporty: `reports/clustering_*.png`, `reports/cluster_profiles.csv`.

## Najlepszy model

Po treningu wybierany jest model z najwyzszym **F1 na zbiorze testowym** (holdout 40%) i zapisywany jako:

- `models/best_model.joblib` — pipeline uzywany w Streamlit
- `reports/best_model.txt` — opis metryk i progu decyzyjnego

Aktualnie: **Voting LR+SVM** (kopiowany z `models/voting_lr_svm.joblib`).

Aktualizacja bez pelnego treningu:
```bash
python scripts/update_best_model.py
```

## Aplikacja Streamlit

Formularz boczny (cechy posortowane wg waznosci RF) → prognoza z najlepszego modelu + porownanie wszystkich 21 modeli z `models/*.joblib`.

```bash
python scripts/run_streamlit.py
```

Aplikacja obsluguje wszystkie typy pipeline'ow:
- ImbPipeline z `engineer`/`preprocess`/`smote`/`scaler`/`model`
- **SoftVotingEnsemble** (voting, voting_lr_svm, voting_top10)
- CatBoost bez kroku `preprocess`
- Warianty top-20 z dodatkowym krokiem `select`

## Zaleznosci kluczowe

- Python 3.11
- `scikit-learn`, `imbalanced-learn` (SMOTE, BalancedRandomForestClassifier)
- `xgboost`, `catboost`
- `tensorflow` 2.21 (Keras 3)
- `streamlit`, `python-pptx`, `joblib`, `pandas`, `matplotlib`, `seaborn`
