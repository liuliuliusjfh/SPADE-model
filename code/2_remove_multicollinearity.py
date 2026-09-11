import pandas as pd; import numpy as np

base = '../data'
train = pd.read_csv(f'{base}/train.csv'); test = pd.read_csv(f'{base}/test.csv')
target = 'Access Control'

X_train = train.drop(columns=[target]); y_train = train[target]
X_test = test.drop(columns=[target]);   y_test = test[target]
print(f'Original features: {len(X_train.columns)}')

# 1. Remove zero-variance features
var = X_train.var()
zero_var = var[var == 0].index.tolist()
X_train.drop(columns=zero_var, inplace=True)
print(f'Removed zero-variance features: {len(zero_var)}')

# 2. Remove features whose most frequent value accounts for more than 95%
n = len(X_train)
high = [c for c in X_train.columns if X_train[c].value_counts().max() / n > 0.95]
X_train.drop(columns=high, inplace=True)
print(f'Removed high-frequency features: {len(high)}')

# 3. Remove highly correlated features, keeping the one more correlated with the target
corr = X_train.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = set()
for col in upper.columns:
    for pair_col in upper[col][upper[col] > 0.98].index:
        if col in to_drop or pair_col in to_drop: continue
        drop = pair_col if abs(y_train.corr(X_train[col])) >= abs(y_train.corr(X_train[pair_col])) else col
        to_drop.add(drop)
X_train.drop(columns=list(to_drop), inplace=True)
print(f'Removed highly correlated features (|r| > 0.98): {len(to_drop)}')

# 4. Keep the same features in the test set and save both datasets
keep = X_train.columns.tolist()
X_test = X_test[keep]
pd.concat([X_train, y_train], axis=1).to_csv(f'{base}/train_clean.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv(f'{base}/test_clean.csv', index=False)
print(f'Final features: {len(keep)} -> {base}/train_clean.csv')
