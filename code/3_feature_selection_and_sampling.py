import pandas as pd; import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb; import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

RS = 42; FOLDS = 5; TARGET = 'Access Control'

SAMPLERS = {
    'No sampling': None, 'SMOTE': SMOTE(random_state=42),
    'Tomek Links': TomekLinks(n_jobs=-1),
    'SMOTE + Tomek': SMOTETomek(random_state=42, n_jobs=-1),
    'Random oversampling': RandomOverSampler(random_state=42),
    'ENN': EditedNearestNeighbours(n_jobs=-1),
    'SMOTE + ENN': SMOTEENN(random_state=42, n_jobs=-1),
}

def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return [roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob),
            accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)]

def ranking(method, X_vals, y_vals):
    if method == 'RFE':
        sel = RFE(xgb.XGBClassifier(random_state=RS, verbosity=0, n_jobs=-1), n_features_to_select=1, step=0.05)
        sel.fit(X_vals, y_vals)
        return np.argsort(sel.ranking_)
    elif method == 'MI':
        return np.argsort(mutual_info_classif(X_vals, y_vals, random_state=RS))[::-1]
    else:
        return np.argsort(np.var(X_vals, axis=0))[::-1]

train = pd.read_csv('../data/train_clean.csv')
X_df = train.drop(columns=[TARGET]).select_dtypes('number').fillna(0)
y = train[TARGET].values
feat_names = X_df.columns.tolist()
X = X_df.values

n_unique = X_df.nunique(); cont_mask = n_unique > 30
print(f'Samples: {len(X)}, features: {X.shape[1]} | Continuous: {cont_mask.sum()}, categorical: {(~cont_mask).sum()}')

def scale_fold(X_tr, X_te, idx):
    cont_in_sel = cont_mask.values[idx]
    cont_local = np.where(cont_in_sel)[0]
    sc = RobustScaler()
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    if len(cont_local) > 0:
        X_tr_s[:, cont_local] = sc.fit_transform(X_tr[:, cont_local])
        X_te_s[:, cont_local] = sc.transform(X_te[:, cont_local])
    return X_tr_s, X_te_s

def eval_stack(X_full, y_full, idx, sampler, kf):
    res = np.zeros((FOLDS, 6))
    for i, (ti, vi) in enumerate(kf.split(X_full)):
        X_tr, y_tr = X_full[ti][:, idx], y_full[ti]
        X_tr, X_te = scale_fold(X_tr, X_full[vi][:, idx], idx)
        y_te = y_full[vi]
        if sampler: X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

        oof_tr = np.zeros((len(X_tr), 3))
        oof_te = np.zeros((len(X_te), 3))
        base_models = [
            xgb.XGBClassifier(n_jobs=-1, random_state=RS, eval_metric='logloss', verbosity=0),
            lgb.LGBMClassifier(n_jobs=-1, random_state=RS, verbose=-1),
            HistGradientBoostingClassifier(random_state=RS),
        ]
        inner_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)
        for j, bm in enumerate(base_models):
            for iti, ivi in inner_kf.split(X_tr):
                bm.fit(X_tr[iti], y_tr[iti])
                oof_tr[ivi, j] = bm.predict_proba(X_tr[ivi])[:, 1]
            bm.fit(X_tr, y_tr)
            oof_te[:, j] = bm.predict_proba(X_te)[:, 1]
        meta = LinearDiscriminantAnalysis()
        meta.fit(oof_tr, y_tr)
        res[i] = metrics(y_te, meta.predict_proba(oof_te)[:, 1])
    return res.mean(axis=0)

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)

for meth in ['RFE', 'MI', 'VT']:
    rk = ranking(meth, X, y)
    print(f'\n{"="*100}')
    print(f'{meth} feature ranking complete; searching for the best k and sampler')
    print(f'{"="*100}')
    print(f'{"Sampler":20s} {"k":>3s} {"AUC":>8s} {"PR-AUC":>8s} {"Acc":>8s} {"Prec":>8s} {"Rec":>8s} {"F1":>8s}')
    print('-' * 72)
    for sn, sp in SAMPLERS.items():
        best_f1, best_k = -1, 0
        best_res = None
        for k in range(40, 61):
            vals = eval_stack(X, y, rk[:k], sp, kf)
            if vals[5] > best_f1:
                best_f1, best_k = vals[5], k
                best_res = vals
        features = [feat_names[i] for i in rk[:best_k]]
        print(f'{sn:20s} {best_k:>3d} {best_res[0]:>8.4f} {best_res[1]:>8.4f} {best_res[2]:>8.4f} {best_res[3]:>8.4f} {best_res[4]:>8.4f} {best_res[5]:>8.4f}')
        print(f'    Features: {features}')
