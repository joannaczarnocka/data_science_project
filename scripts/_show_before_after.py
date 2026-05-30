"""Porownanie wynikow przed i po zmianach anti-overfitting."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import joblib, pandas as pd, numpy as np
from data_loader import load_hr
from features import get_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE
from pathlib import Path

df = load_hr()
X, y, _, _ = get_X_y(df)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_fit, _, y_fit, _ = train_test_split(X_tr, y_tr, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_tr)

SKIP = {'metadata','best_model','preprocessing_pipeline','ensemble'}
SKIP_MODELS = {'baseline', 'stacking', 'stacking_top10'}
SKIP_TOP10 = True   # ukryjmy top10 dla przejrzystosci

def m(pipe, Xs, ys):
    yp  = pipe.predict(Xs)
    ypr = pipe.predict_proba(Xs)[:, 1]
    return round(f1_score(ys, yp, zero_division=0), 4), round(roc_auc_score(ys, ypr), 4)

data = {}
for path in sorted(Path('models').glob('*.joblib')):
    if path.stem in SKIP: continue
    if SKIP_TOP10 and path.stem.endswith('_top10'): continue
    if path.stem in SKIP_MODELS: continue
    try:
        pipe = joblib.load(path)
        f1_train, _ = m(pipe, X_fit, y_fit)
        f1_test, auc = m(pipe, X_te, y_te)
        data[path.stem] = {'f1_train': f1_train, 'f1_test': f1_test, 'auc': auc,
                           'overfit': round(f1_train - f1_test, 4)}
    except Exception as e:
        print(f'SKIP {path.stem}: {e}')

# Wyniki PRZED zmianami (z ostatniego raportu - hardcoded)
before = {
    'logistic_regression': {'f1_train': 0.6897, 'f1_test': 0.5946, 'auc': 0.8236, 'overfit': 0.0951},
    'svm':                 {'f1_train': 0.6886, 'f1_test': 0.5000, 'auc': 0.8138, 'overfit': 0.1886},
    'voting':              {'f1_train': 0.9099, 'f1_test': 0.5314, 'auc': 0.8147, 'overfit': 0.3785},
    'neural_network':      {'f1_train': 0.9697, 'f1_test': 0.5131, 'auc': 0.7945, 'overfit': 0.4566},
    'random_forest':       {'f1_train': 0.6543, 'f1_test': 0.5191, 'auc': 0.7970, 'overfit': 0.1352},
    'balanced_rf':         {'f1_train': 0.6565, 'f1_test': 0.5022, 'auc': 0.7915, 'overfit': 0.1543},
    'xgboost':             {'f1_train': 1.0000, 'f1_test': 0.4565, 'auc': 0.7888, 'overfit': 0.5435},
    'catboost':            {'f1_train': 1.0000, 'f1_test': 0.4880, 'auc': 0.8008, 'overfit': 0.5120},
}

ORDER = sorted(data, key=lambda k: data[k]['f1_test'], reverse=True)

rows = []
for name in ORDER:
    now = data[name]
    was = before.get(name, {})
    rows.append({
        'Model':          name,
        'F1_TRAIN przed': was.get('f1_train', '-'),
        'F1_TRAIN po':    now['f1_train'],
        'F1_TEST przed':  was.get('f1_test', '-'),
        'F1_TEST po':     now['f1_test'],
        'dF1_TEST':       round(now['f1_test'] - was['f1_test'], 4) if was else '-',
        'AUC przed':      was.get('auc', '-'),
        'AUC po':         now['auc'],
        'Overfit przed':  was.get('overfit', '-'),
        'Overfit po':     now['overfit'],
        'dOverfit':       round(now['overfit'] - was['overfit'], 4) if was else '-',
    })

df_out = pd.DataFrame(rows)
pd.set_option('display.width', 160)
pd.set_option('display.max_colwidth', 25)

print('\n' + '='*130)
print('PRZED  vs  PO  — poprawki anti-overfitting')
print('(SMOTE 1.0->0.5 | MLP+SVM: SMOTE->class_weight | XGB/CB/MLP silniejsza reg | SVM C 0.5->0.05)')
print('='*130)
print(df_out.to_string(index=False))
print()
print('Legenda:  dF1_TEST  = F1(po) - F1(przed)  [+ = poprawilo sie]')
print('          dOverfit = Overfit(po) - Overfit(przed)  [- = zmniejszyl sie]')
