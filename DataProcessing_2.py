import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 对训练集和测试集的y按照vid排序,和X一一对应
y_train = pd.read_csv('./dataset/round1_train.csv', encoding='utf-8',
                      low_memory=False, index_col='vid')
y_test = pd.read_csv('./dataset/round1_test.csv', encoding='utf-8',
                     low_memory=False, index_col='vid')
y_train, y_test = y_train.sort_index(), y_test.sort_index()

y_train, y_test = y_train.fillna(0), y_test.fillna(0)
y_train, y_test = y_train.convert_objects(
    convert_numeric=True), y_test.convert_objects(convert_numeric=True)
y_train, y_test = y_train.fillna(0), y_test.fillna(0)
y_train, y_test = y_train.convert_objects(
    convert_numeric=True), y_test.convert_objects(convert_numeric=True)

print(np.isnan(y_train).any())
print(y_train.info(), y_test.info())
y_train.to_csv("./dataset/y_train.csv", encoding='utf-8')
y_test.to_csv("./dataset/y_test.csv", encoding='utf-8')
print("OKAY!!")


print("Start split train and test dataset!!!")
# 读取训练集和测试集
X_train = pd.read_csv('./dataset/train.csv', encoding='utf-8',
                      low_memory=False, header='infer', index_col='vid')
X_test = pd.read_csv('./dataset/test.csv', encoding='utf-8',
                     low_memory=False, header='infer', index_col='vid')
col_train_num, col_test_num = X_train.describe().columns, X_test.describe().columns
X_train,X_test=X_train[col_train_num],X_test[col_test_num]

print("Start filter features!!!")
# 筛选缺失率小于0.1的特征
col_train, col_test = [], []
for item in X_train.columns:
    tmp = np.sum(X_train[item].isnull()) / len(X_train)
    if tmp < 0.95:
        col_train.append(item)
for item in X_test.columns:
    tmp = np.sum(X_test[item].isnull()) / len(X_test)
    if tmp < 0.95:
        col_test.append(item)


# 选择训练集和测试集的交集
col = [item for item in col_train if item in col_test]
print('len(col):', len(col))
X_train, X_test = X_train[col], X_test[col]
print(X_train.shape, X_test.shape)

X_train, X_test = X_train.fillna(0), X_test.fillna(0)
# print(X_train.info(), X_test.info())

#PCA
X_train=PCA(n_components=10).fit_transform(X_train)
X_test=PCA(n_components=10).fit_transform(X_test)

print(X_train.shape, X_test.shape)
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)



X_train,X_test=pd.DataFrame(X_train),pd.DataFrame(X_test)
print(X_train.info(), X_test.info())

X_train.to_csv("./dataset/x_train.csv", encoding='utf-8')
X_test.to_csv("./dataset/x_test.csv", encoding='utf-8')
