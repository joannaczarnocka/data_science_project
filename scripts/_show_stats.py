"""Wypisuje metryki F1, Precision, Recall, AUC, Accuracy na zbiorach TRAIN i TEST
dla wszystkich zapisanych modeli w models/*.joblib (posortowane wg F1_TEST)."""
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
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train)

SKIP = {'metadata', 'best_model', 'preprocessing_pipeline', 'ensemble'}
rows = []
for path in sorted(Path('models').glob('*.joblib')):
    if path.stem in SKIP:
        continue
    try:
        pipe = joblib.load(path)
        for split_name, Xs, ys in [('TRAIN', X_train_fit, y_train_fit), ('TEST', X_test, y_test)]:
            yp  = pipe.predict(Xs)
            ypr = pipe.predict_proba(Xs)[:, 1]
            rows.append({
                'Model':  path.stem,
                'Zbior':  split_name,
                'F1':     round(f1_score(ys, yp, zero_division=0), 4),
                'Prec':   round(precision_score(ys, yp, zero_division=0), 4),
                'Recall': round(recall_score(ys, yp, zero_division=0), 4),
                'AUC':    round(roc_auc_score(ys, ypr), 4),
                'Acc':    round(accuracy_score(ys, yp), 4),
            })
    except Exception as e:
        print(f'SKIP {path.stem}: {e}')

df_res = pd.DataFrame(rows)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 130)
pd.set_option('display.float_format', '{:.4f}'.format)

for split in ['TRAIN', 'TEST']:
    sub = (df_res[df_res['Zbior'] == split]
           .drop(columns='Zbior')
           .sort_values('F1', ascending=False)
           .reset_index(drop=True))
    n = len(X_train_fit) if split == 'TRAIN' else len(X_test)
    print(f'\n{"="*76}')
    print(f'ZBIOR {split}  ({n} probek)')
    print('='*76)
    print(sub.to_string(index=False))
