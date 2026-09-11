import pandas as pd; import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier,
                               BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import TomekLinks
import xgboost as xgb; import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

RS = 42; FOLDS = 5; TARGET = 'Access Control'; K = 46

def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return [roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob),
            accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)]

models = {
    'Stacking': 'stacking',
    'XGBoost': xgb.XGBClassifier(n_jobs=-1, random_state=RS, eval_metric='logloss', verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_jobs=-1, random_state=RS, verbose=-1),
    'HistGB': HistGradientBoostingClassifier(random_state=RS),
    'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=RS),
    'ExtraTrees': ExtraTreesClassifier(n_jobs=-1, random_state=RS),
    'DecisionTree': DecisionTreeClassifier(random_state=RS),
    'Bagging': BaggingClassifier(n_jobs=-1, random_state=RS),
    'GradientB': GradientBoostingClassifier(random_state=RS),
    'AdaBoost': AdaBoostClassifier(random_state=RS),
    'LogisticReg': LogisticRegression(n_jobs=-1, random_state=RS, max_iter=1000),
    'SGD': SGDClassifier(n_jobs=-1, random_state=RS, loss='log_loss'),
    'LDA': LinearDiscriminantAnalysis(),
    'BernoulliNB': BernoulliNB(),
    'MLP': MLPClassifier(random_state=RS),
}

NEEDS_SCALE = {'LogisticReg', 'SGD', 'LDA', 'QDA', 'GaussianNB', 'BernoulliNB', 'MLP'}

train = pd.read_csv('../data/train_clean.csv')
X_df = train.drop(columns=[TARGET]).select_dtypes('number').fillna(0)
y = train[TARGET].values
feat_names = X_df.columns.tolist()
X = X_df.values

# Select the top 46 features with RFE
rk = np.argsort(RFE(xgb.XGBClassifier(random_state=RS, verbosity=0, n_jobs=-1),
                     n_features_to_select=1, step=0.05).fit(X, y).ranking_)
features = [feat_names[i] for i in rk[:K]]
print(f"Total: {len(features)} features")
print(features)

# Evaluate model performance
idx = rk[:K]
n_unique = X_df.nunique(); cont_mask = n_unique > 30
cont_all = np.where(cont_mask)[0]
idx_cont = np.where(cont_mask.values[idx])[0]
sampler = TomekLinks(n_jobs=-1)
scaler = StandardScaler()
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)

print(f'\n{"="*90}')
print(f'{"Model":18s} {"AUC":>8s} {"PR-AUC":>8s} {"Acc":>8s} {"Prec":>8s} {"Rec":>8s} {"F1":>8s}')
print('-' * 72)

for name, model in models.items():
    folds = np.zeros((FOLDS, 6))
    for i, (ti, vi) in enumerate(kf.split(X)):
        X_tr, y_tr = X[ti][:, idx], y[ti]; X_te = X[vi][:, idx].copy(); y_te = y[vi]

        # Apply RobustScaler to continuous features
        sc = RobustScaler()
        X_tr_s = X_tr.copy()
        if len(idx_cont) > 0:
            X_tr_s[:, idx_cont] = sc.fit_transform(X_tr[:, idx_cont])
            X_te[:, idx_cont] = sc.transform(X_te[:, idx_cont])

        # Apply Tomek Links undersampling
        X_tr_s, y_tr = sampler.fit_resample(X_tr_s, y_tr)

        # Apply StandardScaler to linear and Bayesian models
        if name in NEEDS_SCALE:
            ss = StandardScaler()
            X_tr_s[:, idx_cont] = ss.fit_transform(X_tr_s[:, idx_cont])
            X_te[:, idx_cont] = ss.transform(X_te[:, idx_cont])

        if name == 'Stacking':
            base = [
                xgb.XGBClassifier(n_jobs=-1, random_state=RS, eval_metric='logloss', verbosity=0),
                lgb.LGBMClassifier(n_jobs=-1, random_state=RS, verbose=-1),
                HistGradientBoostingClassifier(random_state=RS),
            ]
            ikf = KFold(n_splits=FOLDS, shuffle=True, random_state=RS)
            o_tr, o_te = np.zeros((len(X_tr_s),3)), np.zeros((len(X_te),3))
            for j, bm in enumerate(base):
                for it, iv in ikf.split(X_tr_s):
                    bm.fit(X_tr_s[it], y_tr[it]); o_tr[iv,j] = bm.predict_proba(X_tr_s[iv])[:,1]
                bm.fit(X_tr_s, y_tr); o_te[:,j] = bm.predict_proba(X_te)[:,1]
            prob = LinearDiscriminantAnalysis().fit(o_tr,y_tr).predict_proba(o_te)[:,1]
        else:
            model.fit(X_tr_s, y_tr)
            prob = model.predict_proba(X_te)[:, 1]
        folds[i] = metrics(y_te, prob)
    vals = folds.mean(axis=0)
    print(f'{name:18s} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f} {vals[4]:>8.4f} {vals[5]:>8.4f}')
