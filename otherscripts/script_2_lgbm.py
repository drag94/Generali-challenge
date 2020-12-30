# if __name__ == "__main__":

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import random

test_set = pd.read_csv('/home/davide/Desktop/dati gara generali/test_set.csv', low_memory=False, index_col=False)
train_set = pd.read_csv('/home/davide/Desktop/dati gara generali/train_set.csv', low_memory=False, index_col=False)

# drop the index column
train_set.drop(train_set.columns[0], inplace=True, axis=1)
test_set.drop(test_set.columns[0], inplace=True, axis=1)
# replace field that's entirely == # with NaN
train_set.replace(r'#', np.nan, inplace=True)
test_set.replace(r'#', np.nan, inplace=True)
# copy the test set dataframe into another object
X_test = test_set.copy()
# splitting the target col and feature cols
y_train_raw = train_set['target']
X_train_raw = train_set.drop(train_set.columns[-1], axis=1)
cols_train = X_train_raw.columns

# check for Nans/None values
zip_null = zip(cols_train,
               X_train_raw.isnull().sum(),
               X_test.isnull().sum(),
               )
listanull = list(map(list, zip_null))
Null_table_X = pd.DataFrame(listanull, columns=['featues',
                                                '# of NaNs in X_train_raw',
                                                '# of NaNs in X_test'
                                                ])

print('# of NaNs in the target column: ' + str(y_train_raw.isnull().sum()))  # no nans in the target

# replacing values in the column 13 of X_train such that it is like the column 13 of X_test
X_train_raw['feature_13'] = pd.to_numeric(X_train_raw['feature_13'])
X_test['feature_13'] = pd.to_numeric(X_test['feature_13'])

# convert the strings with their corresponding numeric values and convert to nan non parsing strings
X_train_raw['feature_6'] = pd.to_numeric(X_train_raw['feature_6'], errors='coerce')
X_test['feature_6'] = pd.to_numeric(X_test['feature_6'], errors='coerce')

# In X_train_raw and in X_test, at column 14 the 1s and 0s are strings--> replace them with their corresponding ints
X_train_raw['feature_14'] = pd.to_numeric(X_train_raw['feature_14'])
X_test['feature_14'] = pd.to_numeric(X_test['feature_14'])

print('\nUnique values (train set_13): ' + str(X_train_raw.feature_13.unique()),
      '\nUnique values (train set_14): ' + str(X_train_raw.feature_14.unique()))

print('Unique values (test set_13): ' + str(X_train_raw.feature_14.unique()))

# drop columns with only 1 unique value. Search all columns whose values are True (==1) in Unique_table
Uniques_table = pd.DataFrame([X_train_raw.nunique(), X_test.nunique()], index=['train', 'test'])
uniquecols = Uniques_table[Uniques_table == 1].any()
Trues = uniquecols.index[uniquecols == True].tolist()
X_train_raw.drop(columns=Trues, axis=1, inplace=True)
X_test.drop(columns=Trues, axis=1, inplace=True)

# corr matrix, drop reduntant features
Corr_mat_train = X_train_raw.corr().abs()  # pearson corr
Corr_mat_test = X_test.corr().abs()

columns_train = np.full((Corr_mat_train.shape[0],), True, dtype=bool)
for i in range(Corr_mat_train.shape[0]):
    for j in range(i + 1, Corr_mat_train.shape[0]):
        if Corr_mat_train.iloc[i, j] >= 0.75:
            if columns_train[j]:
                columns_train[j] = False
selected_columns_train = Corr_mat_train.columns[columns_train]

columns_test = np.full((Corr_mat_test.shape[0],), True, dtype=bool)
for i in range(Corr_mat_test.shape[0]):
    for j in range(i + 1, Corr_mat_test.shape[0]):
        if Corr_mat_test.iloc[i, j] >= 0.75:
            if columns_test[j]:
                columns_test[j] = False
selected_columns_test = Corr_mat_test.columns[columns_test]

Noncorrfeatures = list(set(selected_columns_train).intersection(selected_columns_test))

X_train_raw = X_train_raw[Noncorrfeatures]
X_test = X_test[Noncorrfeatures]

# checking for imbalance = 88:12
sns.countplot(y_train_raw)
# plt.show()
target_0 = y_train_raw.value_counts(0)  # 8772-1228
print('\nHow many 0s in the target: ' + str(target_0[0]),
      '\nHow many 1s in the target: ' + str(len(y_train_raw) - target_0[0]),
      '\nratio neg/pos = ' + str(target_0[0] / target_0[1]))

numeric_var_train = [key for key in dict(X_train_raw.dtypes)
                     if dict(X_train_raw.dtypes)[key]
                     in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

cat_var_train = [key for key in dict(X_train_raw.dtypes)
                 if dict(X_train_raw.dtypes)[key] in ['object']]  # dtype O mixed values

print('Columns with numeric values (train set): ' + str(numeric_var_train.__len__()),
      '\n#Columns with non-numeric values (train set): ' + str(cat_var_train.__len__()) +
      '\nwhich features: ' + str(cat_var_train))

numeric_var_test = [key for key in dict(X_test.dtypes)
                    if dict(X_test.dtypes)[key]
                    in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

cat_var_test = [key for key in dict(X_test.dtypes)
                if dict(X_test.dtypes)[key] in ['object']]  # dtype O mixed values

print('# Columns with numeric values (test set): ' + str(numeric_var_test.__len__())
      , '\n# Columns with non-numeric values (test set): ' + str(cat_var_test.__len__()) +
      '\nwhich features: ' + str(cat_var_test))

"SCALING DATA?"
'''It's unnecessary since the base learners are trees, and any monotonic function of any feature variable will have
no effect on how the trees are formed.'''

# how many columns have NANs again.Drop columns with Nan > 4000
nan_dataframe = pd.DataFrame([X_train_raw.isnull().sum(), X_test.isnull().sum()], index=['train', 'test'])
uniquenanscol = nan_dataframe[nan_dataframe >= 4000].any()
Truesnan = uniquenanscol.index[uniquenanscol == True].tolist()
X_train_raw.drop(columns=Truesnan, axis=1, inplace=True)
X_test.drop(columns=Truesnan, axis=1, inplace=True)


def objective(trial):
    train_X, val_x, train_y, val_y = train_test_split(X_train_raw, y_train_raw, test_size=0.25)
    dtrain = lgb.Dataset(train_X, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    params_lgbmodel = {'objective': 'binary',
                       'metric': 'binary_logloss',
                       'scale_pos_weight': [target_0[0] / target_0[1]],
                       'boosting': trial.suggest_categorical('boosting', ['dart', 'rf']),
                       'num_leaves': trial.suggest_int('num_leaves', low=2, high=256),  # per l'overfit
                       'learning_rate': trial.suggest_float('learning_rate', low=0.01, high=0.1, step=0.01),
                       'n_estimators': trial.suggest_int('n_estimators', low=20, high=500, step=15),

                       'min_child_samples': trial.suggest_int('min_child_samples', low=5, high=70),
                       'colsample_bytree': trial.suggest_float('colsample_bytree', low=0.1, high=1, step=0.1),
                       'reg_alpha': trial.suggest_float('reg_alpha', low=0, high=5),  # l1
                       'bagging_fraction': trial.suggest_uniform(name='bagging_fraction', low=0.3, high=1.0),
                       'bagging_freq': trial.suggest_int(name='bagging_freq', low=1, high=7),
                       'feature_fraction': trial.suggest_uniform(name='feature_fraction', low=0.3, high=1.0),
                       'lambda_l1': trial.suggest_loguniform('lambda_l1', low=1e-8, high=15.0),
                       'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 15.0),
                       'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 15),  # per l'overfit
                       'max_delta_step': trial.suggest_int(name='max_delta_step', low=1, high=12),
                       # [1,3,5,7,9,10,11,12],
                       'drop_rate': trial.suggest_float(name='drop_rate', low=.1, high=.5),  # [0.1,0.2,0.3,0.4,0.5],
                       'min_data_in_bin': trial.suggest_int(name='min_data_in_bin', low=10, high=50),
                       }

    # fit and time

    lgbmodel = lgb.train(params_lgbmodel, dtrain, valid_sets=[dval],
                         early_stopping_rounds=100, verbose_eval=False)

    preds = lgbmodel.predict(val_x, num_iteration=lgbmodel.best_iteration)
    pred_labels = np.rint(preds)
    f1 = f1_score(val_y, pred_labels)

    return f1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5000)
    print('Best params : ', study.best_params)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

#############################################################

# 500 trials
# Value: 0.351
#   Params:
#     boosting: rf
#     num_leaves: 247
#     learning_rate: 0.04
#     n_estimators: 369
#     min_child_samples: 8
#     colsample_bytree: 0.8
#     reg_alpha: 3.2589139136441814
#     bagging_fraction: 0.6941812397479091
#     bagging_freq: 2
#     feature_fraction: 0.7433683724699598
#     lambda_l1: 1.3444113768804726e-05
#     lambda_l2: 1.2559231934002553e-05
#     min_data_in_leaf: 13
#     max_delta_step: 7
#     drop_rate: 0.2557515784379373
#     min_data_in_bin: 15

# Number of finished trials: 5000
# Best trial:
#   Value: 0.3671875
#   Params:
#     boosting: dart
#     num_leaves: 27
#     learning_rate: 0.01
#     n_estimators: 290
#     min_child_samples: 52
#     colsample_bytree: 0.6
#     reg_alpha: 2.121067575840617
#     bagging_fraction: 0.6672508287937184
#     bagging_freq: 1
#     feature_fraction: 0.6083195334376993
#     lambda_l1: 0.0012595398095782777
#     lambda_l2: 0.000246505281239431
#     min_data_in_leaf: 11
#     max_delta_step: 4
#     drop_rate: 0.41966285307038387
#     min_data_in_bin: 21
