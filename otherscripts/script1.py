# if __name__ == "__main__":
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import xgboost as xgb
import imblearn
import lightgbm as lgbm
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier #'knn', 'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting', 'linear-svm']
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA,PCA,TruncatedSVD
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import f1_score
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

print('Unique values (train set_13): ' + str(X_train_raw.feature_13.unique())+'\n',
      'Unique values (train set_14): '+ str(X_train_raw.feature_14.unique()))

print('Unique values (test set_13): ' + str(X_train_raw.feature_14.unique()))

#drop columns with only 1 unique value. Search all columns whose values are True (==1) in Unique_table
Uniques_table = pd.DataFrame([X_train_raw.nunique(),X_test.nunique()],index=['train','test'])
uniquecols = Uniques_table[Uniques_table == 1].any()
Trues = uniquecols.index[uniquecols == True].tolist()
X_train_raw.drop(columns=Trues,axis=1,inplace=True)
X_test.drop(columns=Trues,axis=1,inplace=True)

# #corr matrix, drop reduntant features
# Corr_mat_train = X_train_raw.corr().abs() #pearson corr
# Corr_mat_test = X_test.corr().abs()
#
# columns_train = np.full((Corr_mat_train.shape[0],), True, dtype=bool)
# for i in range(Corr_mat_train.shape[0]):
#     for j in range(i+1, Corr_mat_train.shape[0]):
#         if Corr_mat_train.iloc[i,j] >= 0.75:
#             if columns_train[j]:
#                 columns_train[j] = False
# selected_columns_train = Corr_mat_train.columns[columns_train]
#
# columns_test = np.full((Corr_mat_test.shape[0],), True, dtype=bool)
# for i in range(Corr_mat_test.shape[0]):
#     for j in range(i+1, Corr_mat_test.shape[0]):
#         if Corr_mat_test.iloc[i,j] >= 0.75:
#             if columns_test[j]:
#                 columns_test[j] = False
# selected_columns_test = Corr_mat_test.columns[columns_test]
#
# Noncorrfeatures= list(set(selected_columns_train).intersection(selected_columns_test))
#
# X_train_raw = X_train_raw[Noncorrfeatures]
# X_test = X_test[Noncorrfeatures]

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


X_train_raw= pd.get_dummies(X_train_raw,columns=[
                                                           'feature_36',
                                                         'feature_37',
                                                           'feature_38',
                                                          'feature_39'])
X_test=pd.get_dummies(X_test, columns=[
                                                          'feature_36',
                                                          'feature_37',
                                                         'feature_38',
                                                          'feature_39'])


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
#how many columns have NANs again. In order to deal with imbalance and to use Smote, the data needs to be free from nan values
nan_dataframe = pd.DataFrame([X_train_raw.isnull().sum(),X_test.isnull().sum()],index=['train','test'])
uniquenanscol = nan_dataframe[nan_dataframe >= 4000].any()
Truesnan = uniquenanscol.index[uniquenanscol == True].tolist()




#replace missing values with median and not with mean bc median is less sensitive to skewness of the distributions
sgl_imputer = SimpleImputer(missing_values=np.nan, #marks the values that were missing, which might carry some information.
                            strategy='mean'
                                )

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

X_train_imputed_single = pd.DataFrame(data=sgl_imputer.fit_transform(X_train_raw),columns=X_train_raw.columns)
X_test_imputed_single= pd.DataFrame(data=sgl_imputer.fit_transform(X_test),columns=X_test.columns)

######################################################################################################################
######################################################################################################################
######################################################################################################################
count0_target=y_train_raw.value_counts(0)[0]
count1_target = y_train_raw.value_counts(0)[1]
xgbmodel = xgb.XGBClassifier(objective='binary:logistic',  #eval_metric = logloss by default

                              scale_pos_weight= count0_target/count1_target ##useful for unbalanced classes. 8772/1228 = 7.14
                              )
params = {'eta': [0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
    'max_depth': [1,2,3,4,5,6,7,8], #parameter to control overfitting
    'n_estimators': [20,30,40,50,60,100,150,200,250,300,350,400,450,500,550,600,650,700],
    'min_child_weight': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], #parameter to control overfitting
    'gamma':[0,1,1.5,2,3,3.5,4,4.5,5,6,7],              #parameter to control overfitting
    'max_delta_step':[3,4,5,6,7,8,9,10,11,12,13,15,17,19,21,22,23,24,25,27], #1-10 #hadling imbalanced dataset
    'subsample':[0.2,0.3,0.5,0.7,0.8,0.9,1],
    'colsample_bytree':[0.3,0.5,0.7,0.8,1],
    'colsample_bylevel':[0.3,0.5,0.7,0.8,1],
    'colsample_bynode':[0.2,0.1,0.3,0.5,0.7,0.8,1],
    'lambda':[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5],   #l2
    'alpha':[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]    #l1
    }

stratkfold =RepeatedStratifiedKFold(n_splits=5,n_repeats=2,
                                    random_state=10
                                       )

#evaluation
grid = RandomizedSearchCV(xgbmodel,
                          param_distributions=params,
                    scoring='f1',
                    cv=stratkfold,
                    random_state=10, n_jobs=-1
                          )

# fit and time
start_time = timer(None) # timing starts from this point for "start_time" variable
xgb_results_full = grid.fit(X_train_imputed_single, y_train_raw)
timer(start_time) # timing ends here for "start_time" variable

# report the best configuration
print('\n Best estimator: ',grid.best_estimator_)
print('\n Best score: ', grid.best_score_)
print('\n Best parameters: ',grid.best_params_)

###
#0.30 normale senza niente single imputed median
#0.312     normale single imputed mean
#0.308         normale single imputed 0
#0.307          normale knn uniform n=5
# 0.309          normale knn distance n=5
# 0.305  /  0.305      normale knn uniform/distance n=8
# 0.305      /    0.305      normale knn uniform/distance n=3
# 0.302          knn distance n=2

# 0.24            scale_pos=3 single mean
#                   scale_pos = 9 single mean

#0.27 -  0.28   con sparsepca e scalato
#0.27               con scalato e pca normale
#0.27               con pca troncato e scalato
# 0.26        con pca
#          con tpca
#         con spca
#  0.31      con imputed single
# 0.308       con knn imputed
# 0.308       pipeline smote rus
# 0.303        pipeline smoteenn
# 0.14               pipeline smotetomek
# 0.08              pipeline smote only
#               pipeline rus only
# 0.26          con pca e smote rus


###



#plot feature importance
#Get feature importance of each feature. Importance type can be defined as:
#‘weight’: the number of times a feature is used to split the data across all trees.
#‘gain’: the average gain across all splits the feature is used in.
xgb.plot_importance(grid.best_estimator_,max_num_features=30)
plt.show()

#the best features
imp_feat = grid.best_estimator_.get_booster().get_fscore()
best_feat = {k: v for k, v in sorted(imp_feat.items(), key=lambda item: item[1])}
feat_dataframe=pd.DataFrame(best_feat,index=['Fscore_importance']).transpose().sort_values(by='Fscore_importance',ascending=False)
best_feat_all = list(feat_dataframe.index)

#based on the plot and on feat_dataframe as well, i select all the features used to split the data across all trees
X_test_best =pd.DataFrame(data=X_test_imputed_single[best_feat_all],columns=[best_feat_all])
X_train_best =pd.DataFrame(data=X_train_imputed_single[best_feat_all],columns=[best_feat_all])
start_time = timer(None)
grid.fit(X_train_best,y_train_raw)
timer(start_time) # timing ends here for "start_time" variable

print('\n Best estimator: ',grid.best_estimator_)
print('\n Best score: ', grid.best_score_)
print('\n Best parameters: ',grid.best_params_)


results_sel = pd.DataFrame(grid.cv_results_)
#results_sel.to_csv('xgb-grid-search-results-01.csv', index=False)
#LIGHTGBM
lgbmodel = lgbm.LGBMClassifier(objective='binary',
                               is_unbalance=True,
                               max_depth=-1,zero_as_missing=False,
                               )

params_lgbmodel = {'boosting_type':['gbdt','dart','goss','rf'],
                   'num_leaves':[10,15,20,25,30,35,40,45,50],
                   'learning_rate':[0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
                   'n_estimators':[20,30,40,50,60,100,150,200,250,300,350,400,450,500,550,600,650,700],
                    'min_child_samples':[5,10,15,20,25,30],
                   'subsample': [0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1],
                   'colsample_bytree': [0.3, 0.5, 0.7, 0.8, 1],
                   'reg_alpha': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],  # l1
                   'bagging_fraction':[0.3,0.5,0.7,0.9],
                   }

stratkfold =RepeatedStratifiedKFold(n_splits=5,n_repeats=2,
                                    random_state=10
                                       )

#evaluation
grid_lgbm = RandomizedSearchCV(estimator=lgbmodel,
                          param_distributions=params_lgbmodel,
                    scoring='f1',
                    cv=stratkfold,
                    random_state=10, n_jobs=-1
                          )

# fit and time
start_time = timer(None) # timing starts from this point for "start_time" variable
lgbm_result = grid_lgbm.fit(X_train_imputed_single, y_train_raw)
timer(start_time) # timing ends here for "start_time" variable

# report the best configuration
print('\n Best estimator: ',grid_lgbm.best_estimator_)
print('\n Best score: ', grid_lgbm.best_score_)
print('\n Best parameters: ',grid_lgbm.best_params_)

#feature importance
lgbm.plot_importance(grid_lgbm.best_estimator_,importance_type='split',max_num_features=30)
plt.show()
X_train_bestlgbm = pd.DataFrame(X_train_raw,columns=['feature_0','feature_20','feature_268',
                                                     'feature_290',
                                                     'feature_242',
                                                     'feature_2',
                                                     'feature_244',
                                                     'feature_14',
                                                     'feature_22',
                                                     'feature_21'])

pipelineSMOTE_RUS_lgbm = Pipeline([('rus',rus),('smote',smote),('lgbmodel',lgbmodel)])
new_params_smote_rus_lgbm = {'smote__sampling_strategy':[0.6,0.7,0.8,0.9,1,'auto'],
                      'smote__k_neighbors':[1,2,3,4,5,6,7,8,9,10],
'rus__sampling_strategy':[0.7,0.8,0.9,1,'auto']

}
new_params_lgbm  = {'lgbmodel__' + key: params_lgbmodel[key] for key in params_lgbmodel}
new_params_lgbm = dict(new_params_lgbm,**new_params_smote_rus_lgbm)



#based on the plot and on feat_dataframe as well, i select all the features used to split the data across all trees
start_time = timer(None)
grid_lgbm.fit(X_train_imputed_single,y_train_raw)
timer(start_time) # timing ends here for "start_time" variable

print('\n Best estimator: ',grid_lgbm.best_estimator_)
print('\n Best score: ', grid_lgbm.best_score_)
print('\n Best parameters: ',grid_lgbm.best_params_)
#KMeans
# sgl_imputer_kmedia = SimpleImputer(missing_values=np.nan, #marks the values that were missing, which might carry some information.
#                             strategy='median'
#                                 )
# X_train_imputed_single_pca = normpca.fit_transform(X_train_imputed_single)
# #
# normpca = PCA(n_components=10,random_state=10)
# #
# scal = StandardScaler() #avoid breaking the sparsity structure of the data
# X_train_imputed_kmeans = scal.fit_transform(X_train_imputed_single)
#
# X_train_imputed_single_pca = normpca.fit_transform(X_train_imputed_kmeans)
# #
# trunpca = TruncatedSVD(n_components=15,random_state=10)
# X_train_imputed_single_trunc = trunpca.fit_transform((X_train_imputed_kmeans))
#
# kmedia = KMeans(random_state=10)
#
# params_kmeans = {'n_clusters':[2],
#                  'init':['k-means++', 'random'],
#                  'n_init':[1,2,3,4,5,10,15],
#                  'algorithm':['auto','full','elkan']
#
# }
#
# stratkfold =RepeatedStratifiedKFold(n_splits=5,n_repeats=2,
#                                     random_state=10
#                                        )
#
# #evaluation
# grid_kmeans = RandomizedSearchCV(estimator=kmedia,
#                           param_distributions=params_kmeans,
#                     scoring='f1',
#                     cv=stratkfold,
#                     random_state=10
#                           )
#
# # fit and time
# start_time = timer(None) # timing starts from this point for "start_time" variable
# kmedia = grid_kmeans.fit(X_train_imputed_single_trunc, y_train_raw)
# timer(start_time) # timing ends here for "start_time" variable
#
# # report the best configuration
# print('\n Best estimator: ',grid_kmeans.best_estimator_)
# print('\n Best score: ', grid_kmeans.best_score_)
# print('\n Best parameters: ',grid_kmeans.best_params_)
#
#
# #KNN
# knn = KNeighborsClassifier()
#
# params_knn = {
#                  'n_neighbors':[2,3,5,7,9,10,12,14,15],
#                  'weights':['uniform','distance'],
#                  'leaf_size':[10,15,20,25,30,40],
#                  'p':[1,2]
#                  }
#
# stratkfold =RepeatedStratifiedKFold(n_splits=5,n_repeats=2,
#                                     random_state=10
#                                        )
#
# #evaluation
# grid_kmeans = RandomizedSearchCV(estimator=knn,
#                           param_distributions=params_knn,
#                     scoring='f1',
#                     cv=stratkfold,
#                     random_state=10
#                           )
#
# # fit and time
# start_time = timer(None) # timing starts from this point for "start_time" variable
# kmedia = grid_kmeans.fit(X_train_imputed_single_pca, y_train_raw)
# timer(start_time) # timing ends here for "start_time" variable
#
# # report the best configuration
# print('\n Best estimator: ',grid_kmeans.best_estimator_)
# print('\n Best score: ', grid_kmeans.best_score_)
# print('\n Best parameters: ',grid_kmeans.best_params_)




















# ora con le feature migliori facciamo undersampling e oversampling con diverse tecniche
#################################################################################################################################
#Undersampling and oversampling
smote = SMOTE() #vedi doc
bsmote = BorderlineSMOTE()
SVMsmote = SVMSMOTE()
ADASYNsmote = ADASYN()
rus = RandomUnderSampler()
TomekUS = TomekLinks()
Condensednn = CondensedNearestNeighbour()
RepeatednnUS = RepeatedEditedNearestNeighbours()
InstanceUS = InstanceHardnessThreshold()

######################################################################################################################
######################################################################################################################
######################################################################################################################
xgbmodel_smote = xgb.XGBClassifier(objective='binary:logistic'  #eval_metric = logloss by default
                              )

new_params = {'xgbmodel_smote__' + key: params[key] for key in params}
new_params_smote = {'smote__sampling_strategy':[0.2,0.3,0.5,0.7,0.8,0.9,1,'auto'],
                      'smote__k_neighbors':[7,8,9,10,12]}
new_params_smote = dict(new_params,**new_params_smote)

new_params_bsmote = {'bsmote__sampling_strategy':[0.3,0.5,0.7,0.9,'auto'],
                             'bsmote__k_neighbors':[2,3,5,7,10,11,12,13,14],
                         'bsmote__m_neighbors':[3,5,7,10,12,13,14,15]
}
new_params_bsmote = dict(new_params,**new_params_bsmote)

new_params_SVM = {'SVMsmote__sampling_strategy':[0.1,0.2,0.3,0.5,0.7,0.8,0.9,'auto'],
                      'SVMsmote__k_neighbors':[1,2,3,5,7,10]}
new_params_SVM = dict(new_params,**new_params_SVM)

new_params_adasyn = {'ADASYNsmote__sampling_strategy':[0.3,0.5,0.7,0.8,'auto']}
new_params_adasyn = dict(new_params,**new_params_adasyn)

new_params_rus = {'rus__sampling_strategy':[0.3,0.5,0.7,0.8,0.9,1,'auto']}
new_params_rus = dict(new_params,**new_params_rus)

new_params_smote_rus = {'smote__sampling_strategy':[0.6,0.7,0.8,0.9,1,'auto'],
                      'smote__k_neighbors':[1,2,3,4,5,6,7,8,9,10],
'rus__sampling_strategy':[0.7,0.8,0.9,1,'auto']

}
new_params_smote_rus = dict(new_params,**new_params_smote_rus)


new_params_tomek = {'TomekUS__sampling_strategy':['not minority','not majority','all','majority']}
new_params_tomek = dict(new_params,**new_params_tomek)

new_params_condensednn = {'Condensednn__sampling_strategy':['not minority','not majority','all'],
                                        'Condensednn__n_neighbors':[1,2], #size of the neighbourhood to consider to compute the nearest neighbors.
                                       'Condensednn__n_seeds_S':[1,2]}
new_params_condensednn = dict(new_params,**new_params_condensednn)

new_params_repeatednn = {'RepeatednnUS__sampling_strategy':['auto','not majority', 'all'],
                                              'RepeatednnUS__n_neighbors':[3,4,5],
                                               'RepeatednnUS__max_iter':[20,30,40,50,100,150]}
new_params_repeatednn = dict(new_params,**new_params_repeatednn)

new_params_instance = {'InstanceUS__estimator':[xgbmodel_smote],
                                      'InstanceUS__sampling_strategy':[0.1,0.2,0.3,0.4,0.5,0.7,0.8,0.9], #auto
                                        'InstanceUS__cv':[2,3,5,7,8,9]}
new_params_instance = dict(new_params,**new_params_instance)



pipelineSMOTE_RUS = Pipeline([('rus',rus),('smote',smote),('xgbmodel_smote',xgbmodel_smote)])
pipelineBSMOTE_ = Pipeline([('bsmote',bsmote),('xgbmodel_smote',xgbmodel_smote)])
pipelineSVM_ = Pipeline([('SVMsmote',SVMsmote),('xgbmodel_smote',xgbmodel_smote)])
pipelineADASYN_ = Pipeline([('ADASYNsmote',ADASYNsmote),('xgbmodel_smote',xgbmodel_smote)])
pipelineSMOTE_= Pipeline([('smote',smote),('xgbmodel_smote',xgbmodel_smote)])
pipelineRUS = Pipeline([('rus',rus),('xgbmodel_smote',xgbmodel_smote)])
pipelineTomekUS_ = Pipeline([('TomekUS',TomekUS),('xgbmodel_smote',xgbmodel_smote)])
pipelineCondensednn_ = Pipeline([('Condensednn',Condensednn),('xgbmodel_smote',xgbmodel_smote)])
pipelineRepeatednnUS_ = Pipeline([('RepeatednnUS',RepeatednnUS),('xgbmodel_smote',xgbmodel_smote)])
pipelineInstanceUS_ = Pipeline([('InstanceUS',InstanceUS),('xgbmodel_smote',xgbmodel_smote)])
#


stratkfold =RepeatedStratifiedKFold(n_splits=5,n_repeats=2,
                                    random_state=10
                                       )

#evaluation
grid_pipe = RandomizedSearchCV(pipelineSMOTE_RUS_lgbm,
                          param_distributions=new_params_lgbm,
                    scoring='f1',
                    cv=stratkfold,
                    random_state=10, n_jobs=-1
                          )

# fit and time
start_time = timer(None) # timing starts from this point for "start_time" variable
xgb_results_full = grid_pipe.fit(X_train_imputed_single, y_train_raw)
timer(start_time) # timing ends here for "start_time" variable

# report the best configuration
print('\n Best estimator: ',grid_pipe.best_estimator_)
print('\n Best score: ', grid_pipe.best_score_)
print('\n Best parameters: ',grid_pipe.best_params_)

#CON X_train_BEST

#pipelineSMOTE : 0.25
#pipelineSMOTE_RUS(0.9-1-0.7) :0.314
#pipelineBSMOTE_: 0.28
#pipelineSVM_: 0.10
#pipelineADASYN_: 0.21
#pipelineRUS (meglio da solo con >0.6) : 0.303
#pipelineTOMEK : 0.07
#pipelineCONDENSEDNN : troppo tempo
#pipelineREPEATEDNN :
#pipelineINSTANCE : 0.312 -0.310

#
best_xgb = xgb.XGBClassifier(**grid_pipe.best_params_)

## other classifiers
adaboostclf = AdaBoostClassifier()

ada_xgbm = AdaBoostClassifier(xgbmodel)

params_adaboost = {'n_estimators': [20,30,40,50],

                   'learning_rate':[0.001,0.01,0.02,0.03,.04,0.1,0.5,1,1.5,2,3],
                   'algorithm':['SAMME.R']
}
new_params_xgbmodel = {'base_estimator__' + key: params[key] for key in params}
new_params_adaboost = dict(new_params_xgbmodel,**params_adaboost)

stratkfold =RepeatedStratifiedKFold(n_splits=5,n_repeats=2,
                                    random_state=10
                                       )

#evaluation
grid_ada = RandomizedSearchCV(estimator=ada_xgbm,
                          param_distributions=params_adaboost,
                    scoring='f1',
                    cv=stratkfold,
                    random_state=10, n_jobs=-1
                          )

# fit and time
start_time = timer(None) # timing starts from this point for "start_time" variable
xgb_results_full = grid_ada.fit(X_train_imputed_single, y_train_raw)
timer(start_time) # timing ends here for "start_time" variable

# report the best configuration
print('\n Best estimator: ',grid_ada.best_estimator_)
print('\n Best score: ', grid_ada.best_score_)
print('\n Best parameters: ',grid_ada.best_params_)

#################################################################################################
ada_xgbm.feature_importances_()

###############################################################################################(

estimators = [('xgbmodel',xgbmodel),('lgbmodel',lgbmodel)]

stacking = StackingClassifier(estimators=estimators,final_estimator=RandomForestClassifier(),cv=2)

new_params_lgbm = {'lgbmodel__' + key: params_lgbmodel[key] for key in params_lgbmodel}

params_stacking = dict(new_params,**new_params_lgbm)


stratkfold =RepeatedStratifiedKFold(n_splits=5,n_repeats=2,
                                    random_state=10
                                       )

#evaluation
grid_stacking = RandomizedSearchCV(stacking,
                          param_distributions=params_stacking,
                    scoring='f1',
                    cv=stratkfold,
                    random_state=10, n_jobs=-1
                          )

start_time = timer(None) # timing starts from this point for "start_time" variable
grid_stacking.fit(X_train_best,y_train_raw)
timer(start_time) # timing ends here for "start_time" variable

# report the best configuration
print('\n Best estimator: ',grid_stacking.best_estimator_)
print('\n Best score: ', grid_stacking.best_score_)
print('\n Best parameters: ',grid_stacking.best_params_)































#prediction final. X_test is not sampled. So no data leakage
y_pred = grid_smote.predict(X_test)























