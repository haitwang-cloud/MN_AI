import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error


# 读取训练集和测试集
X_train = pd.read_csv('./dataset/x_train.csv', encoding='utf-8', low_memory=False,header='infer',index_col='vid')
X_test = pd.read_csv('./dataset/x_test.csv', encoding='utf-8', low_memory=False,header='infer',index_col='vid')
y_train = pd.read_csv('./dataset/y_train.csv', encoding='utf-8', low_memory=False,header='infer',index_col='vid')
y_test = pd.read_csv('./dataset/y_test.csv', encoding='utf-8', low_memory=False,header='infer',index_col='vid')
# print(X_train.info(), X_test.info())
# print(y_train.info(), y_test.info())
# print(np.isnan(X_train).any())
# print(np.isnan(y_train).any())

print("Start Traing!!!!!!!!!!!!!!!!!!!!!")

y_train_LB, y_train_HB, y_train_TRI, y_train_HDL, y_train_LDL \
    = y_train['LB'], y_train['HB'], y_train['TRI'], y_train['HDL'], y_train['LDL']

clf_LB=KNeighborsRegressor(n_neighbors=80,weights='uniform',n_jobs=-1)
clf_HB=KNeighborsRegressor(n_neighbors=30,weights='uniform',n_jobs=-1)
clf_TRI=KNeighborsRegressor(n_neighbors=95,weights='uniform',n_jobs=-1)
clf_HDL=KNeighborsRegressor(n_neighbors=35,weights='uniform',n_jobs=-1)
clf_LDL=KNeighborsRegressor(n_neighbors=35,weights='uniform',n_jobs=-1)

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


y_LB,y_HB,y_TRI,y_HDL,y_LDL=\
    clf_LB.predict(X_train),clf_HB.predict(X_train),clf_TRI.predict(X_train),\
    clf_HDL.predict(X_train),clf_LDL.predict(X_train)


print("MSE_LB",mean_squared_log_error(y_train_LB,y_LB))
print("MSE_HB",mean_squared_log_error(y_train_HB,y_HB))
print("MSE_TRI",mean_squared_log_error(y_train_TRI,y_TRI))
print("MSE_HDL",mean_squared_log_error(y_train_HDL,y_HDL))
print("MSE_LDL",mean_squared_log_error(y_train_LDL,y_LDL))



result=pd.concat([y_pred_LB,y_pred_HB,y_pred_TRI,y_pred_HDL,y_pred_LDL],axis=1)
result.to_csv("./dataset/PredResultKNN0502.csv", encoding='utf-8',index=False)

# clf=KNeighborsRegressor()
# clf.fit(X_train,y_train_LB)
# k_range=list(np.arange(5,100,5))
# weights_option=['uniform','distance']
# par_grid=dict(n_neighbors=k_range,weights=weights_option)


# grid_knn_1=GridSearchCV(clf,par_grid,cv=10,scoring='neg_mean_squared_log_error',n_jobs=-1,verbose=1)
# grid_knn_1.fit(X_train,y_train_LB)
# print('LB.best_estimator_',grid_knn_1.best_estimator_)
# print('LB.best_score_',grid_knn_1.best_score_)
# print('LB.best_params_',grid_knn_1.best_params_)

# grid_knn_2=GridSearchCV(clf,par_grid,cv=10,scoring='neg_mean_squared_log_error',n_jobs=-1,verbose=1)
# grid_knn_2.fit(X_train,y_train_HB)
# print('HB.best_estimator_',grid_knn_2.best_estimator_)
# print('HB.best_score_',grid_knn_2.best_score_)
# print('HB.best_params_',grid_knn_2.best_params_)

# grid_knn_3=GridSearchCV(clf,par_grid,cv=10,scoring='neg_mean_squared_log_error',n_jobs=-1,verbose=1)
# grid_knn_3.fit(X_train,y_train_TRI)
# print('TRI.best_estimator_',grid_knn_3.best_estimator_)
# print('TRI.best_score_',grid_knn_3.best_score_)
# print('TRI.best_params_',grid_knn_3.best_params_)

# grid_knn_4=GridSearchCV(clf,par_grid,cv=10,scoring='neg_mean_squared_log_error',n_jobs=-1,verbose=1)
# grid_knn_4.fit(X_train,y_train_HDL)
# print('HDL.best_estimator_',grid_knn_4.best_estimator_)
# print('HDL.best_score_',grid_knn_4.best_score_)
# print('HDL.best_params_',grid_knn_4.best_params_)

# grid_knn_5=GridSearchCV(clf,par_grid,cv=10 ,scoring='neg_mean_squared_log_error',n_jobs=-1,verbose=1)
# grid_knn_5.fit(X_train,y_train_LDL)
# print('LDL.best_estimator_',grid_knn_5.best_estimator_)
# print('LDL.best_score_',grid_knn_5.best_score_)
# print('LDL.best_params_',grid_knn_5.best_params_)
