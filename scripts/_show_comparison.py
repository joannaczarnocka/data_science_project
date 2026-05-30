"""Tabela porównawcza: modele pełne (wszystkie cechy) vs warianty top-20 cech per model.
Pokazuje F1_TRAIN, F1_TEST, AUC, Recall, Overfit oraz różnicę dF1_TEST i dOverfit."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import joblib, pandas as pd
from data_loader import load_hr
from features import get_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE
from pathlib import Path

df = load_hr()
X, y, _, _ = get_X_y(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_train_fit, _, y_train_fit, _ = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train)

SKIP = {'metadata', 'best_model', 'preprocessing_pipeline', 'ensemble'}

def metrics(pipe, Xs, ys):
    yp  = pipe.predict(Xs)
    ypr = pipe.predict_proba(Xs)[:, 1]
    return {
        'F1':     round(f1_score(ys, yp, zero_division=0), 4),
        'Prec':   round(precision_score(ys, yp, zero_division=0), 4),
        'Recall': round(recall_score(ys, yp, zero_division=0), 4),
        'AUC':    round(roc_auc_score(ys, ypr), 4),
        'Acc':    round(accuracy_score(ys, yp), 4),
    }

# Wczytaj wszystkie modele
data = {}
for path in sorted(Path('models').glob('*.joblib')):
    if path.stem in SKIP:
        continue
    try:
        pipe = joblib.load(path)
        data[path.stem] = {
            'train': metrics(pipe, X_train_fit, y_train_fit),
            'test':  metrics(pipe, X_test, y_test),
        }
    except Exception as e:
        print(f'SKIP {path.stem}: {e}')

# Rozdziel: bazowe i top10
base_names = [k for k in data if not k.endswith('_top10')]
top_names  = [k for k in data if k.endswith('_top10')]

rows = []
for base in sorted(base_names, key=lambda m: data[m]['test']['F1'], reverse=True):
    top = base + '_top10'
    b_tr = data[base]['train']
    b_te = data[base]['test']
    t_tr = data[top]['train']  if top in data else {}
    t_te = data[top]['test']   if top in data else {}

    delta_train = round(t_tr.get('F1', float('nan')) - b_tr['F1'], 4) if t_tr else None
    delta_test  = round(t_te.get('F1', float('nan')) - b_te['F1'], 4) if t_te else None
    overfit_base = round(b_tr['F1'] - b_te['F1'], 4)
    overfit_top  = round(t_tr.get('F1', float('nan')) - t_te.get('F1', float('nan')), 4) if t_tr and t_te else None

    rows.append({
        'Model':           base,
        'F1_TRAIN':        b_tr['F1'],
        'F1_TRAIN_top20':  t_tr.get('F1', '-') if t_tr else '-',
        'dF1_TRAIN':       delta_train if delta_train is not None else '-',
        'F1_TEST':         b_te['F1'],
        'F1_TEST_top20':   t_te.get('F1', '-') if t_te else '-',
        'dF1_TEST':        delta_test if delta_test is not None else '-',
        'AUC_TEST':        b_te['AUC'],
        'AUC_TEST_top20':  t_te.get('AUC', '-') if t_te else '-',
        'Recall_TEST':     b_te['Recall'],
        'Recall_top20':    t_te.get('Recall', '-') if t_te else '-',
        'overfit_base':    overfit_base,
        'overfit_top20':   overfit_top if overfit_top is not None else '-',
    })

df_out = pd.DataFrame(rows)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 160)
pd.set_option('display.max_colwidth', 30)

print('\n' + '='*130)
print(f'POROWNANIE: MODEL BAZOWY  vs  TOP-20 CECH PER MODEL')
print(f'Zbior treningowy: {len(X_train_fit)} probek  |  testowy: {len(X_test)} probek  |  Attrition=1 w tescie: {int((y_test==1).sum())}')
print('='*130)
print(df_out.to_string(index=False))
print()
print('Legenda:')
print('  dF1_TRAIN / dF1_TEST = F1(top20) - F1(base)  [+ = top20 lepszy]')
print('  overfit = F1_TRAIN - F1_TEST  [im mniejsze tym lepiej]')
