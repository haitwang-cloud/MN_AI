import lightgbm as lgb
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_log_error

# 读取训练集和测试集
X_train = pd.read_csv('./dataset/x_train.csv', encoding='utf-8', low_memory=False,header='infer')
X_test = pd.read_csv('./dataset/x_test.csv', encoding='utf-8', low_memory=False,header='infer')
y_train = pd.read_csv('./dataset/y_train.csv', encoding='utf-8', low_memory=False,header='infer',index_col='vid')
y_test = pd.read_csv('./dataset/y_test.csv', encoding='utf-8', low_memory=False,header='infer',index_col='vid')

y_train, y_test = y_train.fillna(0), y_test.fillna(0)
y_train, y_test = y_train.convert_objects(
    convert_numeric=True), y_test.convert_objects(convert_numeric=True)
# print(X_train.info(), X_test.info())
# print(y_train.info(), y_test.info())
# print(np.isnan(X_train).any())
# print(np.isnan(y_train).any())

print("Start Traing!!!!!!!!!!!!!!!!!!!!!")

y_train_LB, y_train_HB, y_train_TRI, y_train_HDL, y_train_LDL \
    = y_train['LB'], y_train['HB'], y_train['TRI'], y_train['HDL'], y_train['LDL']


clf_LB=lgb.LGBMRegressor(max_depth=10,n_estimators=100,learning_rate=0.01,num_leaves=300,max_bin=200)
clf_HB=lgb.LGBMRegressor(max_depth=10,n_estimators=100,learning_rate=0.01,num_leaves=300,max_bin=200)
clf_TRI=lgb.LGBMRegressor(max_depth=10,n_estimators=100,learning_rate=0.01,num_leaves=300,max_bin=200)
clf_HDL=lgb.LGBMRegressor(max_depth=10,n_estimators=100,learning_rate=0.01,num_leaves=300,max_bin=200)
clf_LDL=lgb.LGBMRegressor(max_depth=10,n_estimators=100,learning_rate=0.01,num_leaves=300,max_bin=200)

clf_LB.fit(X_train,y_train_LB)
clf_HB.fit(X_train,y_train_HB)
clf_TRI.fit(X_train,y_train_TRI)
clf_HDL.fit(X_train,y_train_HDL)
clf_LDL.fit(X_train,y_train_LDL)


y_pred_LB,y_pred_HB,y_pred_TRI,y_pred_HDL,y_pred_LDL=\
    clf_LB.predict(X_test),clf_HB.predict(X_test),clf_TRI.predict(X_test),\
    clf_HDL.predict(X_test),clf_LDL.predict(X_test)
y_pred_LB,y_pred_HB,y_pred_TRI,y_pred_HDL,y_pred_LDL=\
    pd.DataFrame(y_pred_LB),pd.DataFrame(y_pred_HB),pd.DataFrame(y_pred_TRI),\
    pd.DataFrame(y_pred_HDL),pd.DataFrame(y_pred_LDL)
print("okay")

result=pd.concat([y_pred_LB,y_pred_HB,y_pred_TRI,y_pred_HDL,y_pred_LDL],axis=1)
result.to_csv("./dataset/PredResultLGB_0505.csv", encoding='utf-8',index=False)
# 构建LightGBM模型
# model_lgb=lgb.LGBMRegressor()
# param_lgb_list={
#     'boosting':['gbdt'],
#     "max_depth":[10],
#     "n_estimators":[100],
#     "learning_rate":[0.01],
#     "num_leaves":[300],
#     'max_bin':[200],

# }

# grid_lgb_LB=GridSearchCV(model_lgb,param_grid=param_lgb_list,cv=10,verbose=0,n_jobs=-1,scoring='neg_mean_squared_log_error')
# start_time=time.clock()
# grid_lgb_LB.fit(X_train,y_train_LB)
# endtime=time.clock()
# print('LB_best_estimator_',grid_lgb_LB.best_estimator_)
# print('LB_best_score_',grid_lgb_LB.best_score_)
# print('LB_best_params_',grid_lgb_LB.best_params_)
# print("run_time",endtime-start_time)

# scores=cross_val_score(clf_LB,X_train,y_train_LB,cv=10,scoring='neg_mean_squared_log_error')
# print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# grid_lgb_HB=GridSearchCV(model_lgb,param_grid=param_lgb_list,cv=10,verbose=0,n_jobs=-1,scoring='neg_mean_squared_log_error')
# start_time=time.clock()
# X_train.fillna(0)
# y_train_HB.fillna(0)
# grid_lgb_HB.fit(X_train,y_train_HB)
# endtime=time.clock()
# print('LightGBM_best_estimator_',grid_lgb_HB.best_estimator_)
# print('LightGBM_best_score_',grid_lgb_HB.best_score_)
# print('LightGBM_best_params_',grid_lgb_HB.best_params_)
# print("run_time",endtime-start_time)

# grid_lgb_TRI=GridSearchCV(model_lgb,param_grid=param_lgb_list,cv=10,verbose=0,n_jobs=-1,scoring='neg_mean_squared_log_error')
# start_time=time.clock()
# grid_lgb_TRI.fit(X_train,y_train_TRI)
# endtime=time.clock()
# print('LightGBM_best_estimator_',grid_lgb_TRI.best_estimator_)
# print('LightGBM_best_score_',grid_lgb_TRI.best_score_)
# print('LightGBM_best_params_',grid_lgb_TRI.best_params_)
# print("run_time",endtime-start_time)

# grid_lgb_HDL=GridSearchCV(model_lgb,param_grid=param_lgb_list,cv=10,verbose=0,n_jobs=-1,scoring='neg_mean_squared_log_error')
# start_time=time.clock()
# grid_lgb_HDL.fit(X_train,y_train_HDL)
# endtime=time.clock()
# print('LightGBM_best_estimator_',grid_lgb_HDL.best_estimator_)
# print('LightGBM_best_score_',grid_lgb_HDL.best_score_)
# print('LightGBM_best_params_',grid_lgb_HDL.best_params_)
# print("run_time",endtime-start_time)

# grid_lgb_LDL=GridSearchCV(model_lgb,param_grid=param_lgb_list,cv=10,verbose=0,n_jobs=-1,scoring='neg_mean_squared_log_error')
# start_time=time.clock()
# grid_lgb_LDL.fit(X_train,y_train_LDL)
# endtime=time.clock()
# print('LightGBM_best_estimator_',grid_lgb_LDL.best_estimator_)
# print('LightGBM_best_score_',grid_lgb_LDL.best_score_)
# print('LightGBM_best_params_',grid_lgb_LDL.best_params_)
# print("run_time",endtime-start_time)
