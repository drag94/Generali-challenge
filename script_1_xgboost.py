import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

test_set = pd.read_csv('/home/davide/Desktop/dati gara generali/test_set.csv', low_memory=False, index_col=False)
train_set = pd.read_csv('/home/davide/Desktop/dati gara generali/train_set.csv', low_memory=False, index_col=False)


# drop the index column
train_set.drop(train_set.columns[0], inplace=True, axis=1)
test_set.drop(test_set.columns[0], inplace=True, axis=1)
# replace field that's entirely == # with NaN
train_set.replace(r'#',np.nan, inplace=True)
test_set.replace(r'#', np.nan,  inplace=True)
#copy the test set dataframe into another object
X_test = test_set.copy()
# splitting the target col and feature cols
y_train_raw = train_set['target']
X_train_raw = train_set.drop(train_set.columns[-1], axis=1)
cols_train=X_train_raw.columns

# check for Nans/None values
zip_null = zip(cols_train,
               X_train_raw.isnull().sum(),
               X_test.isnull().sum(),
               )
listanull = list(map(list,zip_null))
Null_table_X = pd.DataFrame(listanull, columns=['featues',
                                              '# of NaNs in X_train_raw',
                                              '# of NaNs in X_test'
                                              ])

print('# of NaNs in the target column: ' + str(y_train_raw.isnull().sum())) #no nans in the target

#replacing values in the column 13 of X_train such that it is like the column 13 of X_test
X_train_raw['feature_13'] = pd.to_numeric(X_train_raw['feature_13'])
X_test['feature_13'] = pd.to_numeric(X_test['feature_13'])

#convert the strings with their corresponding numeric values and convert to nan non parsing strings
X_train_raw['feature_6'] = pd.to_numeric(X_train_raw['feature_6'],errors='coerce')
X_test['feature_6'] = pd.to_numeric(X_test['feature_6'],errors='coerce')

#In X_train_raw and in X_test, at column 14 the 1s and 0s are strings--> replace them with their corresponding ints
X_train_raw['feature_14'] = pd.to_numeric(X_train_raw['feature_14'])
X_test['feature_14'] = pd.to_numeric(X_test['feature_14'])

print('\nUnique values (train set_13): ' + str(X_train_raw.feature_13.unique()),
      '\nUnique values (train set_14): '+ str(X_train_raw.feature_14.unique()))

print('Unique values (test set_13): ' + str(X_train_raw.feature_14.unique()))

#drop columns with only 1 unique value. Search all columns whose values are True (==1) in Unique_table
Uniques_table = pd.DataFrame([X_train_raw.nunique(),X_test.nunique()],index=['train','test'])
uniquecols = Uniques_table[Uniques_table == 1].any()
Trues = uniquecols.index[uniquecols == True].tolist()
X_train_raw.drop(columns=Trues,axis=1,inplace=True)
X_test.drop(columns=Trues,axis=1,inplace=True)

#corr matrix, drop reduntant features
Corr_mat_train = X_train_raw.corr().abs() #pearson corr
Corr_mat_test = X_test.corr().abs()

columns_train = np.full((Corr_mat_train.shape[0],), True, dtype=bool)
for i in range(Corr_mat_train.shape[0]):
    for j in range(i+1, Corr_mat_train.shape[0]):
        if Corr_mat_train.iloc[i,j] >= 0.75:
            if columns_train[j]:
                columns_train[j] = False
selected_columns_train = Corr_mat_train.columns[columns_train]

columns_test = np.full((Corr_mat_test.shape[0],), True, dtype=bool)
for i in range(Corr_mat_test.shape[0]):
    for j in range(i+1, Corr_mat_test.shape[0]):
        if Corr_mat_test.iloc[i,j] >= 0.75:
            if columns_test[j]:
                columns_test[j] = False
selected_columns_test = Corr_mat_test.columns[columns_test]

Noncorrfeatures= list(set(selected_columns_train).intersection(selected_columns_test))

X_train_raw = X_train_raw[Noncorrfeatures]
X_test = X_test[Noncorrfeatures]

# checking for imbalance = 88:12
sns.countplot(y_train_raw)
#plt.show()
target_0 = y_train_raw.value_counts(0) #8772-1228
print('\nHow many 0s in the target: ' + str(target_0[0]),
      '\nHow many 1s in the target: ' + str(len(y_train_raw)-target_0[0]),
      '\nratio neg/pos = '+str(target_0[0]/target_0[1]))


numeric_var_train = [key for key in dict(X_train_raw.dtypes)
                     if dict(X_train_raw.dtypes)[key]
                     in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

cat_var_train = [key for key in dict(X_train_raw.dtypes)
                 if dict(X_train_raw.dtypes)[key] in ['object']]  # dtype O mixed values

print('Columns with numeric values (train set): ' + str(numeric_var_train.__len__()),
      '\n#Columns with non-numeric values (train set): ' + str(cat_var_train.__len__()) +
      '\nwhich features: '+str(cat_var_train))


numeric_var_test = [key for key in dict(X_test.dtypes)
               if dict(X_test.dtypes)[key]
               in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

cat_var_test = [key for key in dict(X_test.dtypes)
           if dict(X_test.dtypes)[key] in ['object']]  # dtype O mixed values

print('# Columns with numeric values (test set): ' + str(numeric_var_test.__len__())
      ,'\n# Columns with non-numeric values (test set): ' + str(cat_var_test.__len__()) +
      '\nwhich features: ' + str(cat_var_test))

#how many columns have NANs again.Drop columns with Nan > 4000
nan_dataframe = pd.DataFrame([X_train_raw.isnull().sum(),X_test.isnull().sum()],index=['train','test'])
uniquenanscol = nan_dataframe[nan_dataframe >= 4000].any()
Truesnan = uniquenanscol.index[uniquenanscol == True].tolist()
X_train_raw.drop(columns=Truesnan,axis=1,inplace=True)
X_test.drop(columns=Truesnan,axis=1,inplace=True)

def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X_train_raw,y_train_raw, test_size=0.25,random_state=10)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)
    count0_target = y_train_raw.value_counts(0)[0]
    count1_target = y_train_raw.value_counts(0)[1]

    param = {'objective':'binary:logistic',
              'scale_pos_weight':count0_target/count1_target, ##useful for unbalanced classes. 8772/1228 = 7.14
              'booster': trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            'eta': trial.suggest_float("eta", 1e-8, 1.0, log=True),
              'max_depth': trial.suggest_int('max_depth',1,25,step=1),  # parameter to control overfitting
              'n_estimators':trial.suggest_int('n_estimator',50,500,step=10),
              'min_child_weight':trial.suggest_int('min_child_weight',2,15,step=1), # parameter to control overfitting
              'gamma': trial.suggest_float("gamma", 1e-8, 1.0, log=True), # parameter to control overfitting
              'max_delta_step':trial.suggest_int('max_delta_step',0,15,step=1), # 1-10 #hadling imbalanced dataset
              'subsample':trial.suggest_float('subsample',0,1,step=0.1),
              'colsample_bytree':trial.suggest_float('colsample_bytree',0,1,step=0.1),
              'colsample_bylevel':trial.suggest_float('colsample_bylevel',0,1,step=0.1),
              'colsample_bynode': trial.suggest_float('colsample_bynode',0,1,step=0.05),
              "reg_lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
              "reg_alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
             'random_state': 10
             }


    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain,early_stopping_rounds=100,verbose_eval=False,evals=[(dvalid,'val')])
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    f1 = f1_score(valid_y,pred_labels)
    return f1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5000, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# 300 :0.349
#5000
# Number of finished trials:  2403
# Best trial:
#   Value: 0.3485554520037278
#   Params:
#     booster: gbtree
#     eta: 7.961047489659033e-07
#     max_depth: 4
#     n_estimator: 430
#     min_child_weight: 10
#     gamma: 2.247424300918677e-06
#     max_delta_step: 8
#     subsample: 0.9
#     colsample_bytree: 0.5
#     colsample_bylevel: 0.7000000000000001
#     colsample_bynode: 0.65
#     lambda: 0.00044868464635031266
#     alpha: 2.8228365845205365e-05
#     grow_policy: lossguide








