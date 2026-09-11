import pandas as pd; import numpy as np
import shap; import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.under_sampling import TomekLinks
import xgboost as xgb; import lightgbm as lgb
import os; import warnings; warnings.filterwarnings('ignore')

RS = 42; FOLDS = 5; TARGET = 'Access Control'; K = 46
np.random.seed(RS)
output = '../data/shap_plots'; os.makedirs(output, exist_ok=True)

BEST = {
    'xgb_n': 400, 'xgb_d': 9, 'xgb_lr': 0.2092, 'xgb_ss': 0.8996, 'xgb_cs': 0.7772, 'xgb_mcw': 5,
    'lgb_n': 500, 'lgb_d': 11, 'lgb_lr': 0.2153, 'lgb_ss': 0.8935, 'lgb_cs': 0.8679, 'lgb_nl': 105,
    'hgb_mi': 100, 'hgb_d': 7, 'hgb_lr': 0.2549, 'hgb_msl': 17,
    'lda_solver': 'lsqr',
}

PRE_FEATS = {'Library','Lines','nLines','External_Func','Funds','TryCatch','HashFunc',
    'DelegateCall','Payable_Func','SelfDestruct','Internal_Func','Contracts','Public_Func',
    'avg_FunctionLength','Private_Func','Interfaces','NoOfOutputs','Libraries',
    'NoOfFunctions','LowLevelCall','View_Func','Keccak','Pure_Func','NoOfEvent',
    'avg_InputsOutputs','NoOfInputs','Comment_Lines','Complexity'}

F46 = ['Library','Lines','nLines','External_Func','Funds','TryCatch','HashFunc',
    'DelegateCall','blockNumber','Payable_Func','SelfDestruct','Internal_Func','Contracts',
    'Public_Func','avg_FunctionLength','Private_Func','Block_count','Log_count','d_ORIGIN',
    'mostFrequentOpcode','Interfaces','NoOfOutputs','Libraries','NoOfFunctions','LowLevelCall',
    'View_Func','gas','d_LOG1','d_SWAP10','Stop/Arith','d_CODESIZE','Complexity','Compare_count',
    'Push_count','Keccak','Pure_Func','avg_InputsOutputs','NoOfEvent','d_CODECOPY','d_PUSH20',
    'd_PUSH3','d_PUSH5','NoOfInputs','Comment_Lines','Stack_count','Env_count']
is_pre = np.array([f in PRE_FEATS for f in F46])

# Load the cleaned datasets
train = pd.read_csv('../data/train_clean.csv')
test = pd.read_csv('../data/test_clean.csv')
X_tr_df = train.drop(columns=[TARGET]).select_dtypes('number').fillna(0)
y_tr = train[TARGET].values
X_te_df = test.drop(columns=[TARGET]).select_dtypes('number').fillna(0)
y_te = test[TARGET].values

feat_names_all = X_tr_df.columns.tolist()
idx = np.array([feat_names_all.index(f) for f in F46])

# Extract 46 features and keep raw values for SHAP visualizations
X_tr = X_tr_df.values[:, idx]
X_te_scaled = X_te_df.values[:, idx]         # Used for model training and prediction
X_te_raw = X_te_scaled.copy()                # Raw values used for SHAP visualizations
feat_names = np.array(F46)

cont_mask = X_tr_df.nunique() > 30
idx_cont = np.where(cont_mask.values[idx])[0]
sc = RobustScaler()
if len(idx_cont) > 0:
    X_tr[:, idx_cont] = sc.fit_transform(X_tr[:, idx_cont])
    X_te_scaled[:, idx_cont] = sc.transform(X_te_scaled[:, idx_cont])
X_tr_s, y_tr_s = TomekLinks(n_jobs=-1).fit_resample(X_tr, y_tr)

# Stacking ensemble
base = [
    xgb.XGBClassifier(n_estimators=BEST['xgb_n'], max_depth=BEST['xgb_d'], learning_rate=BEST['xgb_lr'],
        subsample=BEST['xgb_ss'], colsample_bytree=BEST['xgb_cs'], min_child_weight=BEST['xgb_mcw'],
        n_jobs=-1, random_state=RS, eval_metric='logloss', verbosity=0),
    lgb.LGBMClassifier(n_estimators=BEST['lgb_n'], max_depth=BEST['lgb_d'], learning_rate=BEST['lgb_lr'],
        subsample=BEST['lgb_ss'], colsample_bytree=BEST['lgb_cs'], num_leaves=BEST['lgb_nl'],
        n_jobs=-1, random_state=RS, verbose=-1),
    HistGradientBoostingClassifier(max_iter=BEST['hgb_mi'], max_depth=BEST['hgb_d'],
        learning_rate=BEST['hgb_lr'], min_samples_leaf=BEST['hgb_msl'], random_state=RS),
]
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)
oof_tr = np.zeros((len(X_tr_s), 3)); oof_te = np.zeros((len(X_te_scaled), 3))
for j, bm in enumerate(base):
    for ti, vi in kf.split(X_tr_s):
        bm.fit(X_tr_s[ti], y_tr_s[ti]); oof_tr[vi, j] = bm.predict_proba(X_tr_s[vi])[:, 1]
    bm.fit(X_tr_s, y_tr_s); oof_te[:, j] = bm.predict_proba(X_te_scaled)[:, 1]
meta = LinearDiscriminantAnalysis(solver=BEST['lda_solver']).fit(oof_tr, y_tr_s)
yp = meta.predict(oof_te)
print(f'Stacking Test Acc: {np.mean(yp == y_te):.4f}')

def predict_fn(X_np):
    o_tr = np.zeros((len(X_np), 3))
    for j, bm in enumerate(base):
        o_tr[:, j] = bm.predict_proba(X_np)[:, 1]
    return meta.predict_proba(o_tr)[:, 1]

np.random.seed(RS)
bg = X_tr_s[np.random.choice(len(X_tr_s), 100, replace=False)]
explainer = shap.KernelExplainer(predict_fn, bg, feature_names=list(feat_names))
np.random.seed(RS)
shap_te = explainer.shap_values(X_te_scaled, nsamples=100)
print('SHAP computation complete')

# Create three summary plots
shap.summary_plot(shap_te, X_te_raw, feature_names=feat_names, max_display=30, show=False)
plt.tight_layout(); plt.savefig(f'{output}\\(1a)_stacking_top_30_summary.png', dpi=200, bbox_inches='tight'); plt.close()
print('(1a) Top-30 summary plot')

mean_abs = np.abs(shap_te).mean(axis=0); rank = np.argsort(mean_abs)[::-1]
pre_top = rank[is_pre[rank]][:10]; post_top = rank[~is_pre[rank]][:10]

shap.summary_plot(shap_te[:, pre_top], X_te_raw[:, pre_top], feature_names=feat_names[pre_top], show=False)
plt.tight_layout(); plt.savefig(f'{output}\\(1b)_stacking_pre_top_10_summary.png', dpi=200, bbox_inches='tight'); plt.close()
print('(1b) PRE top-10 summary plot')

shap.summary_plot(shap_te[:, post_top], X_te_raw[:, post_top], feature_names=feat_names[post_top], show=False)
plt.tight_layout(); plt.savefig(f'{output}\\(1c)_stacking_post_top_10_summary.png', dpi=200, bbox_inches='tight'); plt.close()
print('(1c) POST top-10 summary plot')

top20 = [f for f in rank][:20]
print(f'Top 20 features: {[feat_names[f] for f in top20]}')
for i, f in enumerate(top20):
    shap.dependence_plot(f, shap_te, X_te_raw, feature_names=feat_names,
                         interaction_index='auto', show=False)
    if feat_names[f] == 'd_LOG1':
        plt.xlim(-0.00005, 0.0025)
    if feat_names[f] == 'd_SWAP10':
        plt.xlim(-0.0001, 0.005)
    if feat_names[f] == 'd_ORIGIN':
        plt.xlim(-0.0001, 0.005)
    plt.tight_layout(); plt.savefig(f'{output}\\(2_{i+1:02d})_{feat_names[f]}_dependence.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f'(2_{i+1:02d}) {feat_names[f]} dependence plot')

# Plot manually selected feature interactions
manual_pairs = [('Libraries', 'DelegateCall'), ('d_LOG1', 'd_SWAP10'),
                ('blockNumber', 'd_ORIGIN'), ('Lines', 'Complexity')]
XLIMS = {'d_LOG1': (-0.00005, 0.0025), 'd_SWAP10': (-0.0001, 0.005), 'd_ORIGIN': (-0.0001, 0.005)}
for k, (f1, f2) in enumerate(manual_pairs):
    i1 = np.where(feat_names == f1)[0][0]
    shap.dependence_plot(i1, shap_te, X_te_raw, feature_names=feat_names,
                         interaction_index=f2, show=False)
    if f1 in XLIMS:
        plt.xlim(*XLIMS[f1])
    plt.tight_layout(); plt.savefig(f'{output}\\(2_{k+21:02d})_{f1}_x_{f2}_interaction.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f'(2_{k+21:02d}) {f1} x {f2} interaction plot')

# Create two waterfall plots
yp_all = meta.predict(oof_te)
tp_i = np.where((yp_all == 1) & (y_te == 1))[0][0]
tn_i = np.where((yp_all == 0) & (y_te == 0))[0][0]
exp_val = explainer.expected_value

shap.waterfall_plot(shap.Explanation(values=shap_te[tp_i], base_values=exp_val,
                                      data=X_te_raw[tp_i], feature_names=feat_names), max_display=10, show=False)
plt.tight_layout(); plt.savefig(f'{output}\\(3a)_stacking_true_positive_waterfall.png', dpi=200, bbox_inches='tight'); plt.close()
print('(3a) True-positive waterfall plot')

shap.waterfall_plot(shap.Explanation(values=shap_te[tn_i], base_values=exp_val,
                                      data=X_te_raw[tn_i], feature_names=feat_names), max_display=10, show=False)
plt.tight_layout(); plt.savefig(f'{output}\\(3b)_stacking_true_negative_waterfall.png', dpi=200, bbox_inches='tight'); plt.close()
print('(3b) True-negative waterfall plot')

print('\nDone!')
