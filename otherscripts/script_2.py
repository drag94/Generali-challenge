# if __name__ == "__main__":
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import xgboost as xgb
import imblearn
import lightgbm as lgb

import optuna
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, RandomForestClassifier
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

# X_train_raw= pd.get_dummies(X_train_raw,columns=[
#                                                            'feature_36',
#                                                          'feature_37',
#                                                            'feature_38',
#                                                           'feature_39'])
# X_test=pd.get_dummies(X_test, columns=[
#                                                           'feature_36',
#                                                           'feature_37',
#                                                          'feature_38',
#                                                           'feature_39'])


#drop columns 6
#X_train_raw.drop(columns='feature_6',inplace=True,axis=1)
#X_test.drop(columns='feature_6',inplace=True,axis=1)

"SCALING DATA?"
'''It's unnecessary since the base learners are trees, and any monotonic function of any feature variable will have
no effect on how the trees are formed.'''

""" SMOTE ##############################################################################################################
One way to solve this problem of imbalanced classification is to oversample the examples in the minority class
An improvement on duplicating examples from the minority class is to synthesize new examples from the minority class.
This is a type of data augmentation for tabular data and can be very effective.
SMOTE first selects a minority class instance a at random and finds its k nearest minority class neighbors.
 The synthetic instance is then created by choosing one of the k nearest neighbors b at random and connecting a and b
 to form a line segment in the feature space. The synthetic instances are generated as a convex combination of the two chosen instances a and b.

The original paper on SMOTE suggested combining SMOTE with random undersampling of the majority class.

The correct application of oversampling during k-fold cross-validation is to apply the method to the training dataset only,
then evaluate the model on the stratified but non-transformed test set.

###############################################################################################
PPCA

Probabilistic PCA which is applicable also on data with missing values. Missing value estimation is typically better
than NIPALS but also slower to compute and uses more memory.
"""
#how many columns have NANs again.Drop columns with Nan > 4000
nan_dataframe = pd.DataFrame([X_train_raw.isnull().sum(),X_test.isnull().sum()],index=['train','test'])
uniquenanscol = nan_dataframe[nan_dataframe >= 4000].any()
Truesnan = uniquenanscol.index[uniquenanscol == True].tolist()
X_train_raw.drop(columns=Truesnan,axis=1,inplace=True)
X_test.drop(columns=Truesnan,axis=1,inplace=True)

#dimensionality reduction con pca
# sgl_imputer = SimpleImputer(missing_values=np.nan, #marks the values that were missing, which might carry some information.
#                             strategy='mean'
#                                 )
#
# scal = StandardScaler() #avoid breaking the sparsity structure of the data
# normpca = PCA(n_components=15,random_state=10)
# sparsepca = SparsePCA(n_components=15,random_state=10, alpha=1,
#                       ridge_alpha=0.01)
#
# trunpca = TruncatedSVD(n_components=15,random_state=10)
#
# kernelpca = KernelPCA(n_components=15,kernel='linear')
# kernelpca = KernelPCA(n_components=15,kernel='poly')
# kernelpca = KernelPCA(n_components=15,kernel='rbf')
# kernelpca = KernelPCA(n_components=15,kernel='sigmoid')
# kernelpca = KernelPCA(n_components=15,kernel='cosine')
#
# X_train_raw_imputed =  pd.DataFrame(data=sgl_imputer.fit_transform(X_train_raw),columns=X_train_raw.columns)
# X_train_raw_norm = pd.DataFrame(data=normalize(X_train_raw_imputed),columns=X_train_raw_imputed.columns)
# X_train_raw_scale = pd.DataFrame(data=scal.fit_transform(X_train_raw_imputed),columns=X_train_raw_imputed.columns)
#
#
# X_train_raw_pca = normpca.fit_transform(X_train_raw_norm)
# X_train_raw_sparsepca = sparsepca.fit_transform(X_train_raw_scale)
# X_train_raw_truncpca =  trunpca.fit_transform((X_train_raw_scale))
#
# sparsepca = SparsePCA(n_components=50,random_state=10,
#                       alpha=1, #Sparsity controlling parameter. Higher values lead to sparser components
#                       ridge_alpha=0.01 #Amount of ridge shrinkage to apply in order to improve conditioning when calling the transform method.
#                       )
# trunpca = TruncatedSVD(n_components=50,random_state=10)
# normpca = PCA(n_components=50,random_state=10)
#
# scal = StandardScaler(with_mean=0) #avoid breaking the sparsity structure of the data
#X_train_imputed_single_scal = scal.fit_transform(X_train_imputed_single)

# X_train_imputed_single_pca = normpca.fit_transform(X_train_imputed_single)
#
# X_train_imputed_single_tpca= trunpca.fit_transform(X_train_imputed_single)
#
# X_train_imputed_single_sppca = sparsepca.fit_transform(X_train_imputed_single)
#







def objective(trial):
    train_X, val_x, train_y, val_y = train_test_split(X_train_raw, y_train_raw, test_size=0.25)
    dtrain = lgb.Dataset(train_X, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    params_lgbmodel = {'objective':'binary',
                       'metric':'binary_logloss',
                       'scale_pos_weight':[target_0[0]/target_0[1]],
                       'boosting': trial.suggest_categorical('boosting',['dart','rf']),
                       'num_leaves': trial.suggest_int('num_leaves', low=2, high=256), #per l'overfit
                       'learning_rate':trial.suggest_float('learning_rate',low=0.01,high=0.1,step=0.01),
                       'n_estimators':trial.suggest_int('n_estimators',low=20,high=500), #[20,30,40,50,60,100,150,200,250,300,350,400,450,500,550,600,650,700],
                       'min_child_samples':trial.suggest_int('min_child_samples',low=5,high=70),
                       'colsample_bytree':trial.suggest_float('colsample_bytree',low=0.1,high=1,step=0.1),
                       'reg_alpha':trial.suggest_float('reg_alpha',low=0,high=5),  # l1
                       'bagging_fraction': trial.suggest_uniform(name='bagging_fraction',low= 0.4,high= 1.0),
                       'bagging_freq': trial.suggest_int(name='bagging_freq',low= 1,high= 7),
                       'feature_fraction': trial.suggest_uniform(name='feature_fraction',low= 0.4,high= 1.0),
                       'lambda_l1': trial.suggest_loguniform('lambda_l1',low= 1e-8,high= 10.0),
                        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                       'min_data_in_leaf':trial.suggest_int('min_data_in_leaf',1,15),#per l'overfit
                       'max_delta_step': trial.suggest_int(name='max_delta_step',low=1,high=12),    #[1,3,5,7,9,10,11,12],
                       'drop_rate':trial.suggest_float(name='drop_rate',low=.1,high=.5), #[0.1,0.2,0.3,0.4,0.5],
                       'min_data_in_bin':trial.suggest_int(name='min_data_in_bin',low=10,high=50),
                       }


    # fit and time

    lgbmodel = lgb.train(params_lgbmodel,dtrain,valid_sets=[dval],
                         early_stopping_rounds=100, verbose_eval=False)

    preds = lgbmodel.predict(val_x, num_iteration=lgbmodel.best_iteration)
    pred_labels = np.rint(preds)
    f1 = f1_score(val_y,pred_labels)

    return f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)
    print('Best params : ',study.best_params)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# con tutto encodato : 0.3447653429602889
# solo 6 numerico e nan>4000 droppati : 0.33665559246954596
# PCA (15) scalato(media=0) imputato (media) : 0.28
# pca (15) normalizzato imputato (media) : 0.27
# 6 numerico correlazione e nan droppati : 0.343
# 500 trials



