import pandas as pd; import numpy as np
import optuna
import multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.under_sampling import TomekLinks
import xgboost as xgb; import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

RS = 42; FOLDS = 5; TARGET = 'Access Control'
K = 46; TRIALS = 200

train = pd.read_csv('../data/train_clean.csv')
X_df = train.drop(columns=[TARGET]).select_dtypes('number').fillna(0)
y = train[TARGET].values; X = X_df.values
feat_names = X_df.columns.tolist()
F46 = ['Library','Lines','nLines','External_Func','Funds','TryCatch','HashFunc',
    'DelegateCall','blockNumber','Payable_Func','SelfDestruct','Internal_Func','Contracts',
    'Public_Func','avg_FunctionLength','Private_Func','Block_count','Log_count','d_ORIGIN',
    'mostFrequentOpcode','Interfaces','NoOfOutputs','Libraries','NoOfFunctions','LowLevelCall',
    'View_Func','gas','d_LOG1','d_SWAP10','Stop/Arith','d_CODESIZE','Complexity','Compare_count',
    'Push_count','Keccak','Pure_Func','avg_InputsOutputs','NoOfEvent','d_CODECOPY','d_PUSH20',
    'd_PUSH3','d_PUSH5','NoOfInputs','Comment_Lines','Stack_count','Env_count']
idx = np.array([feat_names.index(f) for f in F46])
cont_mask = X_df.nunique() > 30

def _run(q, params):
    base_models = [
        xgb.XGBClassifier(n_estimators=params['xgb_n'], max_depth=params['xgb_d'],
            learning_rate=params['xgb_lr'], subsample=params['xgb_ss'],
            colsample_bytree=params['xgb_cs'], min_child_weight=params['xgb_mcw'],
            n_jobs=-1, random_state=RS, eval_metric='logloss', verbosity=0),
        lgb.LGBMClassifier(n_estimators=params['lgb_n'], max_depth=params['lgb_d'],
            learning_rate=params['lgb_lr'], subsample=params['lgb_ss'],
            colsample_bytree=params['lgb_cs'], num_leaves=params['lgb_nl'],
            n_jobs=-1, random_state=RS, verbose=-1),
        HistGradientBoostingClassifier(max_iter=params['hgb_mi'], max_depth=params['hgb_d'],
            learning_rate=params['hgb_lr'], min_samples_leaf=params['hgb_msl'], random_state=RS),
    ]
    sampler = TomekLinks(n_jobs=-1)
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)
    cont_local = np.where(cont_mask.values[idx])[0]

    f1s = []
    for ti, vi in kf.split(X):
        X_tr, y_tr = X[ti][:, idx], y[ti]
        X_tr_s, X_te_s = X_tr.copy(), X[vi][:, idx].copy()
        if len(cont_local) > 0:
            sc = RobustScaler()
            X_tr_s[:, cont_local] = sc.fit_transform(X_tr[:, cont_local])
            X_te_s[:, cont_local] = sc.transform(X_te_s[:, cont_local])
        y_te = y[vi]
        X_tr_s, y_tr = sampler.fit_resample(X_tr_s, y_tr)
        ikf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)
        o_tr, o_te = np.zeros((len(X_tr_s), 3)), np.zeros((len(X_te_s), 3))
        for j, m in enumerate(base_models):
            for it, iv in ikf.split(X_tr_s):
                m.fit(X_tr_s[it], y_tr[it])
                o_tr[iv, j] = m.predict_proba(X_tr_s[iv])[:, 1]
            m.fit(X_tr_s, y_tr)
            o_te[:, j] = m.predict_proba(X_te_s)[:, 1]
        p = LinearDiscriminantAnalysis(solver=params['lda_solver']).fit(o_tr,y_tr).predict_proba(o_te)[:,1]
        f1s.append(f1_score(y_te, p >= 0.5))
    q.put(np.mean(f1s))

def objective(trial):
    params = {
        'xgb_n': trial.suggest_int('xgb_n', 100, 500, step=50),
        'xgb_d': trial.suggest_int('xgb_d', 3, 12),
        'xgb_lr': trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
        'xgb_ss': trial.suggest_float('xgb_ss', 0.6, 1.0),
        'xgb_cs': trial.suggest_float('xgb_cs', 0.6, 1.0),
        'xgb_mcw': trial.suggest_int('xgb_mcw', 1, 10),
        'lgb_n': trial.suggest_int('lgb_n', 100, 500, step=50),
        'lgb_d': trial.suggest_int('lgb_d', 3, 12),
        'lgb_lr': trial.suggest_float('lgb_lr', 0.01, 0.3, log=True),
        'lgb_ss': trial.suggest_float('lgb_ss', 0.6, 1.0),
        'lgb_cs': trial.suggest_float('lgb_cs', 0.6, 1.0),
        'lgb_nl': trial.suggest_int('lgb_nl', 15, 127),
        'hgb_mi': trial.suggest_int('hgb_mi', 100, 500, step=50),
        'hgb_d': trial.suggest_int('hgb_d', 3, 12),
        'hgb_lr': trial.suggest_float('hgb_lr', 0.01, 0.3, log=True),
        'hgb_msl': trial.suggest_int('hgb_msl', 1, 50),
        'lda_solver': trial.suggest_categorical('lda_solver', ['svd', 'lsqr', 'eigen']),
    }
    q = mp.Queue()
    p = mp.Process(target=_run, args=(q, params))
    p.start(); p.join(timeout=600)
    if p.is_alive():
        p.terminate(); p.join()
        print(f'  Trial {trial.number + 1} timed out (10 min)')
        return 0.0
    return q.get() if not q.empty() else 0.0

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
    print(f'\nBest F1: {study.best_value:.4f}')
    print('Best params:', study.best_params)

    bp = study.best_params
    folds = np.zeros((FOLDS, 6))
    for i, (ti, vi) in enumerate(KFold(n_splits=FOLDS, shuffle=True, random_state=RS).split(X)):
        X_tr, y_tr = X[ti][:, idx], y[ti]
        X_tr_s, X_te_s = X_tr.copy(), X[vi][:, idx].copy()
        cl = np.where(cont_mask.values[idx])[0]
        if len(cl) > 0:
            sc = RobustScaler()
            X_tr_s[:, cl] = sc.fit_transform(X_tr[:, cl])
            X_te_s[:, cl] = sc.transform(X_te_s[:, cl])
        y_te = y[vi]
        X_tr_s, y_tr = TomekLinks(n_jobs=-1).fit_resample(X_tr_s, y_tr)
        base_best = [
            xgb.XGBClassifier(n_estimators=bp['xgb_n'], max_depth=bp['xgb_d'], learning_rate=bp['xgb_lr'],
                subsample=bp['xgb_ss'], colsample_bytree=bp['xgb_cs'], min_child_weight=bp['xgb_mcw'],
                n_jobs=-1, random_state=RS, eval_metric='logloss', verbosity=0),
            lgb.LGBMClassifier(n_estimators=bp['lgb_n'], max_depth=bp['lgb_d'], learning_rate=bp['lgb_lr'],
                subsample=bp['lgb_ss'], colsample_bytree=bp['lgb_cs'], num_leaves=bp['lgb_nl'],
                n_jobs=-1, random_state=RS, verbose=-1),
            HistGradientBoostingClassifier(max_iter=bp['hgb_mi'], max_depth=bp['hgb_d'],
                learning_rate=bp['hgb_lr'], min_samples_leaf=bp['hgb_msl'], random_state=RS),
        ]
        ikf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)
        o_tr, o_te = np.zeros((len(X_tr_s), 3)), np.zeros((len(X_te_s), 3))
        for j, m in enumerate(base_best):
            for it, iv in ikf.split(X_tr_s):
                m.fit(X_tr_s[it], y_tr[it])
                o_tr[iv, j] = m.predict_proba(X_tr_s[iv])[:, 1]
            m.fit(X_tr_s, y_tr)
            o_te[:, j] = m.predict_proba(X_te_s)[:, 1]
        p = LinearDiscriminantAnalysis(solver=bp.get('lda_solver', 'svd')).fit(o_tr, y_tr).predict_proba(o_te)[:, 1]
        folds[i] = [roc_auc_score(y_te, p), average_precision_score(y_te, p),
            accuracy_score(y_te, p >= 0.5), precision_score(y_te, p >= 0.5, zero_division=0),
            recall_score(y_te, p >= 0.5, zero_division=0), f1_score(y_te, p >= 0.5, zero_division=0)]
    vals = folds.mean(axis=0)
    print(f'\nBest Stacking: AUC={vals[0]:.4f} PR-AUC={vals[1]:.4f} Acc={vals[2]:.4f} Prec={vals[3]:.4f} Rec={vals[4]:.4f} F1={vals[5]:.4f}')
