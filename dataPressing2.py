import numpy as np
import pandas as pd
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
y_train = y_train.sort_values('vid')
y_test = y_test.sort_values('vid')
y_train.to_csv("./dataset/y_train.csv", encoding='utf-8', index=False)
y_test.to_csv("./dataset/y_test.csv", encoding='utf-8', index=False)
print("sort Y OK!")
"""
X_train = pd.read_csv('./dataset/S_train.csv', encoding='utf-8', low_memory=False)
X_test = pd.read_csv('./dataset/S_test.csv', encoding='utf-8', low_memory=False)
y_train=pd.read_csv('./dataset/y_train.csv', encoding='utf-8', low_memory=False)[:200]
y_test=pd.read_csv('./dataset/y_test.csv', encoding='utf-8', low_memory=False)[:200]
"""筛选缺失率小于0.5的特征"""
# print(X_train.columns[1:])
# print(np.sum(X_train['0102'].isnull()) / len(X_train))
# count=[]
col_train,col_test=[],[]
for item in X_train.columns[1:]:
    tmp=np.sum(X_train[item].isnull())/len(X_train)
    if tmp<0.2:
        col_train.append(item)
for item in X_test.columns[1:]:
    tmp=np.sum(X_test[item].isnull())/len(X_test)
    if tmp<0.2:
        col_test.append(item)
#选择训练集和测试集的交集
col=[item for item in col_train if item in col_test]
print(len(col))
print(X_train[col][:2])