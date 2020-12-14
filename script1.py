# if __name__ == "__main__":
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score


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
cols_train = X_train_raw.columns
# check for Nans/None values
zip_null = zip(cols_train,
               X_train_raw.isnull().sum(),
               X_test.isnull().sum(),
               )
listanull = list(map(list,zip_null))
Null_table_X = pd.DataFrame(listanull, columns=['featues',
                                              '# of NaNs in X_train',
                                              '# of NaNs in X_test'
                                              ])
print('# of NaNs in the target column: ' + str(y_train_raw.isnull().sum())) #no nans in the target

# checking for imbalance = 88:12
sns.countplot(y_train_raw)
plt.show()
target_0 = y_train_raw.value_counts(0) #8772-1228
print('How many 0s in the target: ' + str(target_0[0]) + "\n",
      'How many 1s in the target: ' + str(len(y_train_raw)-target_0[0]))


#replacing values in the column 13 of X_train such that it is like the column 13 of X_test
X_train_raw.replace(['01','02','03','04','05','06','07','08','09','00','10','11','12','13','14','15','16','17','18'],
                                                [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18],
                                                inplace=True,regex=True)

#X_train_raw.fillna(-111.0,inplace=True)
#X_test.fillna(-111.0,inplace=True)

#In X_train_raw and in X_test, at column 14 the 1s and 0s are strings--> replace them with their corresponding ints
X_train_raw.replace(['1','0'],[1.0,0.0],inplace=True,regex=True)
X_test.replace(['1','0'],[1.0,0.0],inplace=True,regex=True)
print('Unique values (train set_13): ' + str(X_train_raw.feature_13.unique())+'\n',
      'Unique values (train set_14): '+ str(X_train_raw.feature_14.unique()))

print('Unique values (test set_13): ' + str(X_train_raw.feature_14.unique()))

#drop column: feature_6,36,37,38,39 because i think they're useless because they have a lot of missing values
X_train_raw.drop(X_train_raw.columns[[6,36,37,38,39]],inplace=True,axis=1)
X_test.drop(X_test.columns[[6,36,37,38,39]],inplace=True,axis=1)


# checking for numeric and non-num columns
numeric_var_train = [key for key in dict(X_train_raw.dtypes)
                     if dict(X_train_raw.dtypes)[key]
                     in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

cat_var_train = [key for key in dict(X_train_raw.dtypes)
                 if dict(X_train_raw.dtypes)[key] in ['object']]  # Categorical Variable ---> feature 13-14 dtype O mixed values

print('Columns with numeric values (train set): ' + str(numeric_var_train.__len__()),
      "\n"+'Columns with non-numeric values (train set): ' + str(cat_var_train.__len__()) +"\n"+
      'what features: '+str(cat_var_train))


numeric_var_test = [key for key in dict(X_test.dtypes)
               if dict(X_test.dtypes)[key]
               in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

cat_var_test = [key for key in dict(X_test.dtypes)
           if dict(X_test.dtypes)[key] in ['object']]  # Categorical Variable --> feature 14 dtype O mixed values

print('# Columns with numeric values (test set): ' + str(numeric_var_test.__len__())
      ,"\n"+'# Columns with non-numeric values (test set): ' + str(cat_var_test.__len__()) + "\n"+
      'what features: ' + str(cat_var_test))

Uniques_table = pd.DataFrame([X_train_raw.nunique(),X_test.nunique()],index=['train','test'])
#drop columns with only 1 unique value. Search all columns whose values == 0 and drop them
col_0_train = X_train_raw.columns[(X_train_raw == 0.0).all()]
col_0_test = X_test.columns[(X_test == 0.0).all()]
X_train_raw.drop(col_0_train,axis=1,
                 inplace=True)
X_test.drop(col_0_test,inplace=True,axis=1)
#drop columns where values are almost all 0s (only 2 unique values) both train and test set
mer = Uniques_table[Uniques_table == 2].any()
print('columns where values are almost all 0s (only 2 unique values) both train and test set: ',list(mer[mer].index))
#['feature_2', 'feature_3', 'feature_5', 'feature_18', 'feature_25', 'feature_32', 'feature_46', 'feature_47', 'feature_49',
# 'feature_50', 'feature_62', 'feature_63', 'feature_130', 'feature_147', 'feature_163', 'feature_201', 'feature_217', 'feature_235']
# 2 18 25 46 47 49 50 163 201 217 235
"""
X_train_raw.drop(['feature_2','feature_18','feature_25','feature_46','feature_47','feature_49','feature_50',
                  'feature_163','feature_201','feature_217','feature_235'],axis=1,inplace=True)
X_test.drop(['feature_2','feature_18','feature_25','feature_46','feature_47','feature_49','feature_50',
                  'feature_163','feature_201','feature_217','feature_235'],axis=1,inplace=True)
"""

"SCALING DATA?"
'''It's unnecessary since the base learners are trees, and any monotonic function of any feature variable will have 
no effect on how the trees are formed.'''

print('X_train shape: ',X_train_raw.shape, '\n' +
       'y_train shape: ',y_train_raw.shape,'\n' +
        'X_test shape: ', X_test.shape)


######################################################################################################################
######################################################################################################################
######################################################################################################################
xgb_model = xgb.XGBClassifier(objective='binary:logistic',  #eval_metric = logloss by default
                              #missing=-111.0,
                              scale_pos_weight=7.14, ##useful for unbalanced classes. 7.14 = 8772/1228 = 0 instances / 1 instances in the target array
                                            )
                            #tree_methood=hist
params = {'eta': [0.01,0.02,0.03,0.04,0.05,0.1],
    'max_depth': [1,2,3,4,5,6,7,8], #parameter to control overfitting
    'n_estimators': [40,50,60,80,100,150,200,300,350,400,500],
    'min_child_weight': [0,1,2,3,4,5], #parameter to control overfitting
    'gamma':[0,0.5,1,1.5,2,2.5,3],              #parameter to control overfitting
    'max_delta_step':[0,1,2,3,4,5,6,7,8,9,10], #1-10 #hadling imbalanced dataset
    'subsample':[0.1,0.2,0.3,0.5,0.7,0.9,1],
    'colsample_bytree':[0.3,0.5,0.7,0.8,1],
    'colsample_bylevel':[0.3,0.5,0.7,0.8,1],
    'colsample_bynode':[0.3,0.5,0.7,0.8,1],
    'lambda':[0.5,1,1.5,2,2.5,3,3.5],   #l2
    'alpha':[0,0.5,1,1.5,2],    #l1
    }
stratkfold =StratifiedKFold(n_splits=5,
                            shuffle=True
                                       )

grid = RandomizedSearchCV(xgb_model,
                    param_distributions=params,
                    scoring='f1',
                    cv=stratkfold,
                    random_state=10, n_jobs=-1
                          )
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
xgb_results = grid.fit(X_train_raw,y_train_raw)
timer(start_time) # timing ends here for "start_time" variable

# report the best configuration
print('\n Best estimator: ',grid.best_estimator_)
print('\n Best score: ', grid.best_score_)
print('\n Best parameters: ',grid.best_params_)

results = pd.DataFrame(grid.cv_results_)

#plot feature importance
xgb.plot_importance(grid.best_estimator_,max_num_features=40)
plt.show()
#based on the plot, i select the most important features
X_train_sel = X_train_raw[['feature_20','feature_268','feature_290','feature_34','feature_0','feature_74','feature_242','feature_78',
                           'feature_294','feature_54','feature_244','feature_14','feature_83','feature_75','feature_247','feature_28',
                           'feature_241','feature_19','feature_11','feature_72','feature_61','feature_15','feature_279','feature_132',
                        'feature_85','feature_10']]

xgb_model_sel=grid.fit(X_train_sel,y_train_raw)

print('\n Best estimator: ',grid.best_estimator_)
print('\n Best score: ', grid.best_score_)
print('\n Best parameters: ',grid.best_params_)
# Fit model using each importance as a threshold
#thresholds = sort()
results.to_csv('xgb-grid-search-results-01.csv', index=False)
#Prediction array
y_pred_xgb = xgb_model.predict(test_set)

















