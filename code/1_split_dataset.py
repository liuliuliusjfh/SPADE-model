import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/data.csv')
labels_df = pd.read_csv('../data/label.csv')
df = df.merge(labels_df, left_on='contractAddress', right_on='contractID', how='inner')

target = 'Access Control'
drop_cols = ['contractAddress', 'contractID', 'quarter', 'part_of_day', 'methodId',
             'Reentrancy', 'Arithmetic', 'Unchecked Return Values', 'DoS',
             'Bad Randomness', 'Front Running', 'Time manipulation']
X = df.drop(columns=drop_cols + [target]).select_dtypes('number').fillna(0)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

base = '../data'
import os; os.makedirs(base, exist_ok=True)
pd.concat([X_train, y_train], axis=1).to_csv(f'{base}/train.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv(f'{base}/test.csv', index=False)
print(f'Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(X.columns)}')
