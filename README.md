## 美年健康AI大赛个人代码(初赛排名220)

# 运行顺序：
1. DataProcessing_1.py 用来进行数据预处理，生成label|fea1|fea2格式的数据集train.csv和test.csv
2. DataProcessing_2.py用来筛选train.csv和test.csv中共有的缺失率小于0.95的特征（暂时只选用了数字特征，未采用文本特征），对特征进行PCA降维，接着进行缩放生成X_train.csv,X_test.csv,y_train.csv,y_test.csv
3. LightGBM.py和KNN.py是分别利用LightGBM和KNN进行建模预测的代码