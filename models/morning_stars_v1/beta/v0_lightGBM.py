import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder


# File paths
test_path = '../../../../../data/splitted_datasets/allele/beta/test.tsv'
train_path = '../../../../../data/splitted_datasets/allele/beta/train.tsv'
valid_path = '../../../../../data/splitted_datasets/allele/beta/validation.tsv'

# Load the TSV files
train_df = pd.read_csv(train_path, sep='\t')
valid_df = pd.read_csv(valid_path, sep='\t', low_memory=False)
test_df = pd.read_csv(test_path, sep='\t')

# # Define columns
# feature_cols = ['TRB_CDR3', 'Epitope', 'TRBV', 'TRBJ', 'MHC']
# target_col = 'Binding'

# # Label encode all features (basic encoding for simplicity)
# encoders = {}
# for col in feature_cols:
#     le = LabelEncoder()
#     all_data = pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0)
#     le.fit(all_data.astype(str))
#     train_df[col] = le.transform(train_df[col].astype(str))
#     valid_df[col] = le.transform(valid_df[col].astype(str))
#     test_df[col] = le.transform(test_df[col].astype(str))
#     encoders[col] = le

# # Identify categorical columns (LightGBM handles them natively)
# categorical_features = ['TRBV', 'TRBJ', 'MHC']

# # LightGBM datasets
# train_data = lgb.Dataset(train_df[feature_cols], label=train_df[target_col], categorical_feature=categorical_features)
# valid_data = lgb.Dataset(valid_df[feature_cols], label=valid_df[target_col], reference=train_data, categorical_feature=categorical_features)

# # LightGBM parameters
# params = {
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'boosting_type': 'gbdt',
#     'verbosity': -1,
#     'seed': 42
# }

# # Train the model
# model = lgb.train(
#     params,
#     train_data,
#     valid_sets=[train_data, valid_data],
#     valid_names=['train', 'val'],
#     num_boost_round=1000,
#     # early_stopping_rounds=20
# )
# # Predict probabilities
# y_pred = model.predict(test_df[feature_cols])

# # Convert to binary predictions
# y_pred_binary = (y_pred > 0.5).astype(int)

# ===================================
# model = LGBMClassifier(
#     objective='binary',
#     boosting_type='gbdt',
#     n_estimators=1000,
#     random_state=42
# )


# model.fit(
#     train_df[feature_cols],
#     train_df[target_col],
#     eval_set=[(valid_df[feature_cols], valid_df[target_col])],
#     eval_metric='binary_logloss',
#     # early_stopping_rounds=20,
#     categorical_feature=categorical_features
#     # verbose=50
# )
# # Predict probabilities
# y_pred = model.predict_proba(test_df[feature_cols])[:, 1]
# y_pred_binary = (y_pred > 0.5).astype(int)
#==========================================

# y_true = test_df[target_col]
# print('Accuracy:', accuracy_score(y_true, y_pred_binary))
# print('AUC:', roc_auc_score(y_true, y_pred))
# print('F1 Score:', f1_score(y_true, y_pred_binary))
# print('AP Score:', average_precision_score(y_true, y_pred))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


seq_cols = ['TRB_CDR3', 'Epitope']
encoders = {}

for col in seq_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], valid_df[col], test_df[col]]).astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    valid_df[col] = le.transform(valid_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    encoders[col] = le

from lightgbm import LGBMClassifier

feature_cols = ['TRB_CDR3', 'Epitope', 'TRBV', 'TRBJ', 'MHC']
X_train = train_df[feature_cols]
X_valid = valid_df[feature_cols]
X_test = test_df[feature_cols]
y_train = train_df['Binding']
y_valid = valid_df['Binding']
y_test = test_df['Binding']

model = LGBMClassifier(n_estimators=1000, random_state=42)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='binary_logloss',
    callbacks=[early_stopping(20), log_evaluation(50)],
    categorical_feature=['TRBV', 'TRBJ', 'MHC']
)

# Predict probabilities
y_pred = model.predict_proba(test_df[feature_cols])[:, 1]
y_pred_binary = (y_pred > 0.5).astype(int)
==========================================

y_true = test_df[target_col]
print('Accuracy:', accuracy_score(y_true, y_pred_binary))
print('AUC:', roc_auc_score(y_true, y_pred))
print('F1 Score:', f1_score(y_true, y_pred_binary))
print('AP Score:', average_precision_score(y_true, y_pred))

# print("using OneHotEncoder:")


# # Initialize encoder
# ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 

# # Fit on combined TCR + Epitope from train, then transform all
# ohe.fit(train_df[['TRB_CDR3', 'Epitope']])

# # Transform each set
# train_seq = ohe.transform(train_df[['TRB_CDR3', 'Epitope']])
# valid_seq = ohe.transform(valid_df[['TRB_CDR3', 'Epitope']])
# test_seq  = ohe.transform(test_df[['TRB_CDR3', 'Epitope']])

# # For categorical features (TRBV, TRBJ, MHC)

# cat_cols = ['TRBV', 'TRBJ', 'MHC']
# for col in cat_cols:
#     le = LabelEncoder()
#     le.fit(pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0).astype(str))
#     train_df[col] = le.transform(train_df[col].astype(str))
#     valid_df[col] = le.transform(valid_df[col].astype(str))
#     test_df[col] = le.transform(test_df[col].astype(str))

# # Concatenate the one-hot and label-encoded features
# import numpy as np

# X_train = np.hstack([train_seq, train_df[cat_cols].values])
# X_valid = np.hstack([valid_seq, valid_df[cat_cols].values])
# X_test  = np.hstack([test_seq, test_df[cat_cols].values])

# y_train = train_df['Binding']
# y_valid = valid_df['Binding']
# y_test  = test_df['Binding']

# model = LGBMClassifier(n_estimators=1000, random_state=42)

# model.fit(
#     X_train, y_train,
#     eval_set=[(X_valid, y_valid)],
#     eval_metric='binary_logloss',
#     callbacks=[early_stopping(20), log_evaluation(50)]
# )