"""Wyniki TRAIN i TEST: modele pelne vs warianty top-20 cech."""
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
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_fit, _, y_fit, _ = train_test_split(X_tr, y_tr, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_tr)

SKIP = {'metadata', 'best_model', 'preprocessing_pipeline', 'ensemble', 'baseline', 'stacking', 'stacking_top10'}

def row(name, pipe):
    def m(Xs, ys):
        yp  = pipe.predict(Xs)
        ypr = pipe.predict_proba(Xs)[:, 1]
        return (round(f1_score(ys, yp, zero_division=0), 4),
                round(precision_score(ys, yp, zero_division=0), 4),
                round(recall_score(ys, yp, zero_division=0), 4),
                round(roc_auc_score(ys, ypr), 4),
                round(accuracy_score(ys, yp), 4))
    f_tr, p_tr, r_tr, a_tr, ac_tr = m(X_fit, y_fit)
    f_te, p_te, r_te, a_te, ac_te = m(X_te,  y_te)
    return {'model': name,
            'F1_tr': f_tr, 'Prec_tr': p_tr, 'Rec_tr': r_tr,
            'F1_te': f_te, 'Prec_te': p_te, 'Rec_te': r_te, 'AUC_te': a_te, 'Acc_te': ac_te,
            'overfit': round(f_tr - f_te, 4)}

base_rows, top_rows = [], []
for path in sorted(Path('models').glob('*.joblib')):
    if path.stem in SKIP: continue
    try:
        pipe = joblib.load(path)
        r = row(path.stem, pipe)
        (top_rows if path.stem.endswith('_top10') else base_rows).append(r)
    except Exception as e:
        print(f'SKIP {path.stem}: {e}')

df_base = pd.DataFrame(base_rows).sort_values('F1_te', ascending=False).reset_index(drop=True)
df_top  = pd.DataFrame(top_rows).sort_values('F1_te', ascending=False).reset_index(drop=True)

pd.set_option('display.width', 140)
pd.set_option('display.float_format', '{:.4f}'.format)

SEP = '=' * 110

print(f'\n{SEP}')
print(f'WSZYSTKIE CECHY (modele pelne)  |  TRAIN={len(X_fit)}  TEST={len(X_te)}  Attrition=1: {int((y_te==1).sum())}')
print(SEP)
print(df_base.to_string(index=False))

print(f'\n{SEP}')
print(f'TOP-20 CECH PER MODEL (z ograniczeniem kolumn)')
print(SEP)
print(df_top.to_string(index=False))

# Zestawienie F1_TEST: base vs top20
print(f'\n{SEP}')
print('ZESTAWIENIE F1_TEST: pelne cechy vs top-20')
print(SEP)
base_map = {r['model']: r for r in base_rows}
top_map  = {r['model'].replace('_top10', ''): r for r in top_rows}
compare = []
for bname, br in sorted(base_map.items(), key=lambda x: x[1]['F1_te'], reverse=True):
    tr = top_map.get(bname)
    compare.append({
        'Model':        bname,
        'F1_tr (full)': br['F1_tr'],
        'F1_te (full)': br['F1_te'],
        'F1_tr (top20)': tr['F1_tr'] if tr else '-',
        'F1_te (top20)': tr['F1_te'] if tr else '-',
        'dF1_te':        round(tr['F1_te'] - br['F1_te'], 4) if tr else '-',
        'Overfit full':  br['overfit'],
        'Overfit top20': tr['overfit'] if tr else '-',
        'dOverfit':      round((tr['overfit'] - br['overfit']), 4) if tr else '-',
    })
df_cmp = pd.DataFrame(compare)
print(df_cmp.to_string(index=False))
print('\n  dF1_te  > 0  => top-20 lepszy na tescie')
print('  dOverfit < 0  => top-20 mniej overfituje')
