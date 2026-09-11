import pandas as pd; import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.under_sampling import TomekLinks
import xgboost as xgb; import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

RS = 42; FOLDS = 5; TARGET = 'Access Control'
K = 46

# Best hyperparameters

BEST = {
    'xgb_n': 400, 'xgb_d': 9, 'xgb_lr': 0.2092, 'xgb_ss': 0.8996, 'xgb_cs': 0.7772, 'xgb_mcw': 5,
    'lgb_n': 500, 'lgb_d': 11, 'lgb_lr': 0.2153, 'lgb_ss': 0.8935, 'lgb_cs': 0.8679, 'lgb_nl': 105,
    'hgb_mi': 100, 'hgb_d': 7, 'hgb_lr': 0.2549, 'hgb_msl': 17,
    'lda_solver': 'lsqr',
}

F46 = ['Library','Lines','nLines','External_Func','Funds','TryCatch','HashFunc',
    'DelegateCall','blockNumber','Payable_Func','SelfDestruct','Internal_Func','Contracts',
    'Public_Func','avg_FunctionLength','Private_Func','Block_count','Log_count','d_ORIGIN',
    'mostFrequentOpcode','Interfaces','NoOfOutputs','Libraries','NoOfFunctions','LowLevelCall',
    'View_Func','gas','d_LOG1','d_SWAP10','Stop/Arith','d_CODESIZE','Complexity','Compare_count',
    'Push_count','Keccak','Pure_Func','avg_InputsOutputs','NoOfEvent','d_CODECOPY','d_PUSH20',
    'd_PUSH3','d_PUSH5','NoOfInputs','Comment_Lines','Stack_count','Env_count']

train = pd.read_csv('../data/train_clean.csv')
test = pd.read_csv('../data/test_clean.csv')

X_tr_full = train.drop(columns=[TARGET]).select_dtypes('number').fillna(0)
y_tr_full = train[TARGET].values
X_te = test.drop(columns=[TARGET]).select_dtypes('number').fillna(0)
y_te = test[TARGET].values

feat_names = X_tr_full.columns.tolist()
idx = np.array([feat_names.index(f) for f in F46])

X_tr = X_tr_full.values[:, idx]
X_te = X_te.values[:, idx]
cont_mask = X_tr_full.nunique() > 30
cont_local = np.where(cont_mask.values[idx])[0]

# Scale continuous features
sc = RobustScaler()
if len(cont_local) > 0:
    X_tr[:, cont_local] = sc.fit_transform(X_tr[:, cont_local])
    X_te[:, cont_local] = sc.transform(X_te[:, cont_local])

# Apply Tomek Links undersampling to the training set
tl = TomekLinks(n_jobs=-1)
X_tr_s, y_tr_s = tl.fit_resample(X_tr, y_tr_full)
print(f'TomekLinks: {len(X_tr)} → {len(X_tr_s)}')

# Train the stacking ensemble
base_models = [
    xgb.XGBClassifier(n_estimators=BEST['xgb_n'], max_depth=BEST['xgb_d'], learning_rate=BEST['xgb_lr'],
        subsample=BEST['xgb_ss'], colsample_bytree=BEST['xgb_cs'], min_child_weight=BEST['xgb_mcw'],
        n_jobs=-1, random_state=RS, eval_metric='logloss', verbosity=0),
    lgb.LGBMClassifier(n_estimators=BEST['lgb_n'], max_depth=BEST['lgb_d'], learning_rate=BEST['lgb_lr'],
        subsample=BEST['lgb_ss'], colsample_bytree=BEST['lgb_cs'], num_leaves=BEST['lgb_nl'],
        n_jobs=-1, random_state=RS, verbose=-1),
    HistGradientBoostingClassifier(max_iter=BEST['hgb_mi'], max_depth=BEST['hgb_d'],
        learning_rate=BEST['hgb_lr'], min_samples_leaf=BEST['hgb_msl'], random_state=RS),
]

# Generate out-of-fold predictions
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)
oof_tr = np.zeros((len(X_tr_s), 3))
oof_te = np.zeros((len(X_te), 3))
for j, bm in enumerate(base_models):
    for ti, vi in kf.split(X_tr_s):
        bm.fit(X_tr_s[ti], y_tr_s[ti])
        oof_tr[vi, j] = bm.predict_proba(X_tr_s[vi])[:, 1]
    bm.fit(X_tr_s, y_tr_s)
    oof_te[:, j] = bm.predict_proba(X_te)[:, 1]

meta = LinearDiscriminantAnalysis(solver=BEST['lda_solver'])
meta.fit(oof_tr, y_tr_s)
prob = meta.predict_proba(oof_te)[:, 1]

print('\nTest set performance:')
print(f'  AUC:      {roc_auc_score(y_te, prob):.4f}')
print(f'  PR-AUC:   {average_precision_score(y_te, prob):.4f}')
print(f'  Acc:      {accuracy_score(y_te, prob >= 0.5):.4f}')
print(f'  Prec:     {precision_score(y_te, prob >= 0.5, zero_division=0):.4f}')
print(f'  Rec:      {recall_score(y_te, prob >= 0.5, zero_division=0):.4f}')
print(f'  F1:       {f1_score(y_te, prob >= 0.5, zero_division=0):.4f}')
