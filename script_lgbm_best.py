import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import random
import os
import random
if __name__ == '__main__':



    input_test_set_path = '/home/davide/Desktop/dati gara generali/test_set.csv'
    input_train_set_path = '/home/davide/Desktop/dati gara generali/train_set.csv'

    test_set = pd.read_csv(input_test_set_path, low_memory=False, index_col=False)
    train_set = pd.read_csv(input_train_set_path, low_memory=False, index_col=False)

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
                   X_test.isnull().sum())
    listanull = list(map(list, zip_null))
    Null_table_X = pd.DataFrame(listanull, columns=['featues',
                                                    '# of NaNs in X_train_raw',
                                                    '# of NaNs in X_test'])

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

    # corr matrix, drop redundant features
    Corr_mat_train = X_train_raw.corr().abs()  # pearson corr
    Corr_mat_test = X_test.corr().abs()

    # dropping correlated features
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

    noncorr_features = list(set(selected_columns_train).intersection(selected_columns_test))

    X_train_raw = X_train_raw[noncorr_features]
    X_test = X_test[noncorr_features]

    # checking for imbalance = 88:12
    sns.countplot(y_train_raw)
    #plt.show()
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

    # how many columns have NANs again.Drop columns with Nan > 4000
    nan_dataframe = pd.DataFrame([X_train_raw.isnull().sum(), X_test.isnull().sum()], index=['train', 'test'])
    uniquenanscol = nan_dataframe[nan_dataframe >= 4000].any()
    Truesnan = uniquenanscol.index[uniquenanscol == True].tolist()
    X_train_raw.drop(columns=Truesnan, axis=1, inplace=True)
    X_test.drop(columns=Truesnan, axis=1, inplace=True)

    opt_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'scale_pos_weight': [target_0[0] / target_0[1]],
        'boosting': 'dart',
        'num_leaves': 27,
        'learning_rate': 0.01,
        'n_estimators': 290,
        'min_child_samples': 52,
        'colsample_bytree': 0.6,
        'reg_alpha': 2.121067575840617,
        'bagging_fraction': 0.6672508287937184,
        'bagging_freq': 1,
        'feature_fraction': 0.6083195334376993,
        'lambda_l1': 0.0012595398095782777,
        'lambda_l2': 0.000246505281239431,
        'min_data_in_leaf': 11,
        'max_delta_step': 4,
        'drop_rate': 0.41966285307038387,
        'min_data_in_bin': 21
    }

    train_X, val_x, train_y, val_y = train_test_split(X_train_raw, y_train_raw, test_size=0.25, random_state=10)
    dtrain = lgb.Dataset(train_X, label=train_y)

    lgbmodel = lgb.LGBMClassifier(random_state=10) #lgb.cv(opt_params, dtrain)
    lgbmodel.set_params(**opt_params)
    lgbmodel.fit(X_train_raw, y_train_raw)
    preds_test = lgbmodel.predict(X_test)
    pred_labels = np.rint(preds_test)
    np.savetxt("test_predictions_lgb.csv", pred_labels, delimiter=",")











