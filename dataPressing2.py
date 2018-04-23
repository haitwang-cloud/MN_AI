import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
"""
# 首先生成小样本数据集【训练集和测试集都是200条数据】进行特征提取
# 读取训练集和测试集
train = pd.read_csv('./dataset/train.csv', encoding='utf-8', low_memory=False)
test = pd.read_csv('./dataset/test.csv', encoding='utf-8', low_memory=False)
# 选取各200条作为小样本集合
S_train, S_test = train[:200], test[:200]
print(S_train.shape, S_test.shape)
S_train.to_csv("./dataset/S_train.csv", encoding='utf-8', index=False)
S_test.to_csv("./dataset/S_test.csv", encoding='utf-8', index=False)
print("S-dataset OK!")

#对训练集和测试集的y按照vid排序,和X一一对应
y_train = pd.read_csv('./dataset/round1_train.csv', encoding='utf-8', low_memory=False)
y_test = pd.read_csv('./dataset/round1_test.csv', encoding='utf-8', low_memory=False)
y_train=y_train.convert_objects(convert_numeric=True)
print(y_train.info(),y_test.info())
y_train = y_train.sort_values('vid')
y_test = y_test.sort_values('vid')
y_train.to_csv("./dataset/y_train.csv", encoding='utf-8', index=False)
y_test.to_csv("./dataset/y_test.csv", encoding='utf-8', index=False)
print("sort Y OK!")
"""
X_train = pd.read_csv('./dataset/S_train.csv', encoding='utf-8', low_memory=False,header='infer')
X_test = pd.read_csv('./dataset/S_test.csv', encoding='utf-8', low_memory=False,header='infer')
y_train = pd.read_csv('./dataset/y_train.csv', encoding='utf-8', low_memory=False,header='infer')[:200]
y_test = pd.read_csv('./dataset/y_test.csv', encoding='utf-8', low_memory=False,header='infer')[:200]
y_train, y_test = y_train[y_train.columns[1:]], y_test[y_test.columns[1:]]
print(y_train.info(),y_test.info())

"""筛选缺失率小于0.5的特征"""
col_train, col_test = [], []
for item in X_train.columns[:]:
    tmp = np.sum(X_train[item].isnull()) / len(X_train)
    if tmp < 0.3:
        col_train.append(item)
for item in X_test.columns[:]:
    tmp = np.sum(X_test[item].isnull()) / len(X_test)
    if tmp < 0.3:
        col_test.append(item)
# 选择训练集和测试集的交集
col = [item for item in col_train if item in col_test]
print('len(col):', len(col))
X_train, X_test = X_train[col], X_test[col]
print(X_train.shape, X_test.shape)
# 筛选数值行列
col_num = [item for item in col if item in X_train.describe().columns]

X_train, X_test = X_train[col_num], X_test[col_num]
X_train,X_test=X_train.fillna(0),X_test.fillna(0)
y_train,y_test=y_train.fillna(0),y_test.fillna(0)
print(X_train.shape, X_test.shape)

y_train_LB, y_train_HB, y_train_TRI, y_train_HDL, y_train_LDL \
    = y_train['LB'], y_train['HB'], y_train['TRI'], y_train['HDL'], y_train['LDL']

clf=KNeighborsRegressor()
clf.fit(X_train,y_train_LB)
k_range=list(np.arange(5,10))
weights_option=['uniform','distance']
par_grid=dict(n_neighbors=k_range,weights=weights_option)


grid_knn_1=GridSearchCV(clf,par_grid,cv=5,scoring='r2')
grid_knn_1.fit(X_train,y_train_LB)
print('LB.best_estimator_',grid_knn_1.best_estimator_)
print('LB.best_score_',grid_knn_1.best_score_)
print('LB.best_params_',grid_knn_1.best_params_)

grid_knn_2=GridSearchCV(clf,par_grid,cv=5,scoring='r2')
grid_knn_2.fit(X_train,y_train_HB)
print('HB.best_estimator_',grid_knn_2.best_estimator_)
print('HB.best_score_',grid_knn_2.best_score_)
print('HB.best_params_',grid_knn_2.best_params_)

grid_knn_3=GridSearchCV(clf,par_grid,cv=5,scoring='r2')
grid_knn_3.fit(X_train,y_train_TRI)
print('TRI.best_estimator_',grid_knn_3.best_estimator_)
print('TRI.best_score_',grid_knn_3.best_score_)
print('TRI.best_params_',grid_knn_3.best_params_)

grid_knn_4=GridSearchCV(clf,par_grid,cv=5,scoring='r2')
grid_knn_4.fit(X_train,y_train_HDL)
print('HDL.best_estimator_',grid_knn_4.best_estimator_)
print('HDL.best_score_',grid_knn_4.best_score_)
print('HDL.best_params_',grid_knn_4.best_params_)

grid_knn_5=GridSearchCV(clf,par_grid,cv=5,scoring='r2')
grid_knn_5.fit(X_train,y_train_LDL)
print('LDL.best_estimator_',grid_knn_5.best_estimator_)
print('LDL.best_score_',grid_knn_5.best_score_)
print('LDL.best_params_',grid_knn_5.best_params_)