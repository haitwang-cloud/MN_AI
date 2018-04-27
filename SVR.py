import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

#对训练集和测试集的y按照vid排序,和X一一对应
y_train = pd.read_csv('./dataset/round1_train.csv', encoding='utf-8', low_memory=False)
y_test = pd.read_csv('./dataset/round1_test.csv', encoding='utf-8', low_memory=False)
y_train = y_train.sort_values('vid')
y_test = y_test.sort_values('vid')

y_train.to_csv("./dataset/y_train.csv", encoding='utf-8',index=False)
y_test.to_csv("./dataset/y_test.csv", encoding='utf-8',index=False)
print("Start split train and test dataset!!!")
# 读取训练集和测试集
X_train = pd.read_csv('./dataset/train.csv', encoding='utf-8', low_memory=False,header='infer')
X_test = pd.read_csv('./dataset/test.csv', encoding='utf-8', low_memory=False,header='infer')
y_train = pd.read_csv('./dataset/y_train.csv', encoding='utf-8', low_memory=False,header='infer')
y_test = pd.read_csv('./dataset/y_test.csv', encoding='utf-8', low_memory=False,header='infer')

X_train,X_test=X_train.set_index('vid'),X_test.set_index('vid')
# y_train,y_test=y_train.set_index('vid'),y_test.set_index('vid')
y_train=y_train.convert_objects(convert_numeric=True)
y_test=y_test.convert_objects(convert_numeric=True)
print(y_train.info(),y_test.info())

print(X_train.shape, X_test.shape)
# print(y_train.info(),y_test.info())
print("Start filter features!!!")
"""筛选缺失率小于0.1的特征"""
col_train, col_test = [], []
for item in X_train.columns[:]:
    tmp = np.sum(X_train[item].isnull()) / len(X_train)
    if tmp < 0.1:
        col_train.append(item)
for item in X_test.columns[:]:
    tmp = np.sum(X_test[item].isnull()) / len(X_test)
    if tmp < 0.1:
        col_test.append(item)
# 选择训练集和测试集的交集
col = [item for item in col_train if item in col_test]
print('len(col):', len(col))
X_train, X_test = X_train[col], X_test[col]
print(X_train.shape, X_test.shape)

X_train,X_test=X_train.fillna(0),X_test.fillna(0)

X_train=X_train.convert_objects(convert_numeric=True)
X_test=X_test.convert_objects(convert_numeric=True)

X_train,X_test=X_train.fillna(0),X_test.fillna(0)
y_train,y_test=y_train.fillna(0),y_test.fillna(0)

X_train=X_train.convert_objects(convert_numeric=True)
X_test=X_test.convert_objects(convert_numeric=True)

print(X_train.info(),X_test.info())

# X_train.to_csv("./dataset/tmp.csv", encoding='utf-8', index=False)
print("Start Traing!!!!!!!!!!!!!!!!!!!!!")

y_train_LB, y_train_HB, y_train_TRI, y_train_HDL, y_train_LDL \
    = y_train['LB'], y_train['HB'], y_train['TRI'], y_train['HDL'], y_train['LDL']

# clf_LB=KNeighborsRegressor(n_neighbors=16,weights='distance',n_jobs=-1)
# clf_HB=KNeighborsRegressor(n_neighbors=16,weights='uniform',n_jobs=-1)
# clf_TRI=KNeighborsRegressor(n_neighbors=16,weights='uniform',n_jobs=-1)
# clf_HDL=KNeighborsRegressor(n_neighbors=16,weights='uniform',n_jobs=-1)
# clf_LDL=KNeighborsRegressor(n_neighbors=16,weights='uniform',n_jobs=-1)
#
# clf_LB.fit(X_train,y_train_LB)
# clf_HB.fit(X_train,y_train_HB)
# clf_TRI.fit(X_train,y_train_TRI)
# clf_HDL.fit(X_train,y_train_HDL)
# clf_LDL.fit(X_train,y_train_LDL)
#
# y_pred_LB,y_pred_HB,y_pred_TRI,y_pred_HDL,y_pred_LDL=\
#     clf_LB.predict(X_test),clf_HB.predict(X_test),clf_TRI.predict(X_test),\
#     clf_HDL.predict(X_test),clf_LDL.predict(X_test)
# y_pred_LB,y_pred_HB,y_pred_TRI,y_pred_HDL,y_pred_LDL=\
#     pd.DataFrame(y_pred_LB),pd.DataFrame(y_pred_HB),pd.DataFrame(y_pred_TRI),\
#     pd.DataFrame(y_pred_HDL),pd.DataFrame(y_pred_LDL)
# print(type(y_pred_LB))
# y_train_LB.to_csv("./dataset/tmp1.csv", encoding='utf-8')
# y_train_HB.to_csv("./dataset/tmp2.csv", encoding='utf-8')
# y_train_TRI.to_csv("./dataset/tmp3.csv", encoding='utf-8')
# y_train_HDL.to_csv("./dataset/tmp4.csv", encoding='utf-8')
# y_train_LDL.to_csv("./dataset/tmp5.csv", encoding='utf-8')
# result=pd.concat([y_pred_LB,y_pred_HB,y_pred_TRI,y_pred_HDL,y_pred_LDL],axis=1)
# result.to_csv("./dataset/PredResult.csv", encoding='utf-8',index=False)

clf=SVR()
kernel_option=['linear','rbf','sigmoid']
par_grid=dict(kernel=kernel_option)


grid_knn_1=GridSearchCV(clf,par_grid,cv=5,scoring='r2',n_jobs=-1,verbose=1)
grid_knn_1.fit(X_train,y_train_LB)
print('LB.best_estimator_',grid_knn_1.best_estimator_)
print('LB.best_score_',grid_knn_1.best_score_)
print('LB.best_params_',grid_knn_1.best_params_)

grid_knn_2=GridSearchCV(clf,par_grid,cv=5,scoring='r2',n_jobs=-1,verbose=1)
grid_knn_2.fit(X_train,y_train_HB)
print('HB.best_estimator_',grid_knn_2.best_estimator_)
print('HB.best_score_',grid_knn_2.best_score_)
print('HB.best_params_',grid_knn_2.best_params_)

grid_knn_3=GridSearchCV(clf,par_grid,cv=5,scoring='r2',n_jobs=-1,verbose=1)
grid_knn_3.fit(X_train,y_train_TRI)
print('TRI.best_estimator_',grid_knn_3.best_estimator_)
print('TRI.best_score_',grid_knn_3.best_score_)
print('TRI.best_params_',grid_knn_3.best_params_)

grid_knn_4=GridSearchCV(clf,par_grid,cv=5,scoring='r2',n_jobs=-1,verbose=1)
grid_knn_4.fit(X_train,y_train_HDL)
print('HDL.best_estimator_',grid_knn_4.best_estimator_)
print('HDL.best_score_',grid_knn_4.best_score_)
print('HDL.best_params_',grid_knn_4.best_params_)

grid_knn_5=GridSearchCV(clf,par_grid,cv=5,scoring='r2',n_jobs=-1,verbose=1)
grid_knn_5.fit(X_train,y_train_LDL)
print('LDL.best_estimator_',grid_knn_5.best_estimator_)
print('LDL.best_score_',grid_knn_5.best_score_)
print('LDL.best_params_',grid_knn_5.best_params_)