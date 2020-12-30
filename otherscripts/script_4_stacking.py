from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import xgboost as xgb
import imblearn
import lightgbm as lgb
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier #'knn', 'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting', 'linear-svm']
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import SparsePCA,PCA,TruncatedSVD, KernelPCA
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE, BorderlineSMOTE,SVMSMOTE,ADASYN
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours, TomekLinks,\
    CondensedNearestNeighbour, InstanceHardnessThreshold


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


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
plt.show()
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

"SCALING DATA?"
'''It's unnecessary since the base learners are trees, and any monotonic function of any feature variable will have
no effect on how the trees are formed.'''

#how many columns have NANs again.Drop columns with Nan > 4000
nan_dataframe = pd.DataFrame([X_train_raw.isnull().sum(),X_test.isnull().sum()],index=['train','test'])
uniquenanscol = nan_dataframe[nan_dataframe >= 4000].any()
Truesnan = uniquenanscol.index[uniquenanscol == True].tolist()
X_train_raw.drop(columns=Truesnan,axis=1,inplace=True)
X_test.drop(columns=Truesnan,axis=1,inplace=True)
######################################################################################################################

lgbmodel = lgb.LGBMClassifier(boosting= 'rf',
    num_leaves= 247,
    learning_rate= 0.04,
    n_estimators= 369,
    min_child_samples= 8,
    colsample_bytree= 0.8,
    reg_alpha= 3.2589139136441814,
    bagging_fraction= 0.6941812397479091,
    bagging_freq= 2,
    feature_fraction= 0.7433683724699598,
    lambda_l1= 1.3444113768804726e-05,
    lambda_l2= 1.2559231934002553e-05,
    min_data_in_leaf= 13,
    max_delta_step= 7,
    drop_rate= 0.2557515784379373,
    min_data_in_bin= 15,random_state=10)

xgbmodel = xgb.XGBClassifier(booster='gbtree',
    eta= 7.961047489659033e-07,
    max_depth= 4,
    n_estimator= 430,
    min_child_weight= 10,
    gamma= 2.247424300918677e-06,
    max_delta_step= 8,
    subsample= 0.9,
    colsample_bytree= 0.5,
    colsample_bylevel= 0.7000000000000001,
    colsample_bynode= 0.65,
    reg_lambda= 0.00044868464635031266,
    alpha= 2.8228365845205365e-05,
    grow_policy= 'lossguide',random_state=10)

######################################################################################################################
#nello stacking metto i parametri migliori trovati singolarmente























