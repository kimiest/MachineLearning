import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm

'''
逻辑回归算法手工复现， 使用mnist数据集
mnist手写体识别数据集格式：[[3, 28*28个元素], [], []]
'''

np.seterr(divide='ignore',invalid='ignore')

# *******************
# 加载数据集
# *******************
def load_data(file_path):
    print('开始加载数据')
    features = []  # 存放样本特征的列表
    labels = []  # 存放样本标签的列表
    data = pd.read_csv(file_path)  # 此时data是pandas.DataFrame类型的数据，类似于excel表格，有表头和行号
    data_list = data.values.tolist()  # DataFrame -> 二维列表
    for item in data_list:  # 循环读取每一个样本实例
        L = 0 if item[0] < 5 else 1  # 用于二分类任务，需要将10个标签映射到[0,1]两个标签上
        labels.append(L)
        features.append([F/255 for F in item[1:]])  # 对特征进行归一化处理
    return features, labels  # features=[[28*28个元素],[],[]...] labels=[3, 6, 7, ...]


# ******************
# 逻辑回归训练函数
# ******************
def logist_regression(features, labels, epoch=50, lr=0.0001):
    num_data = len(labels)  # 样本数量
    for i in range(num_data):
        features[i].append(1)  # 给特征增加一个维度，取值为1，主要用于计算偏置
    feature_dim = len(features[0])   # 特征维度
    w = np.zeros(feature_dim)   # 初始化权重向量

    for e in tqdm(range(epoch)):  # 在全部训练样本上迭代一轮
        for i in range(num_data):  # 对每个样本进行梯度下降，更新参数
            yi = np.array(labels[i])  # 把标签和特征列表都转换为numpy数组，以进行向量或矩阵运算
            xi = np.array(features[i])
            wx = np.dot(w, xi)
            # 更新参数
            w += lr * (xi * yi - (np.exp(wx) * xi) / (1 + np.exp(wx)))

    return w


# **************
# 预测一个样本
# **************
def predict(w, x):
    wx = np.dot(w, x)  # 向量点积
    P1 = np.exp(wx) / (1 + np.exp(wx))  # sigmoid函数
    if P1 >= 0.5:
        return 1
    else:
        return 0


# ********************
# 模型预测函数
# *****************
def model_test(features, labels, w):
    # 与训练过程一致，特征增加一个维度，取值为1
    num_data = len(labels)
    for i in range(num_data):
        features[i].append(1)

    # 错误计数
    error = 0
    for i in tqdm(range(num_data)):  # 对测试集中每个样本进行测试
        # 如果真实标签与预测标签不同，则error+=1
        if np.array(labels[i]) != predict(w, np.array(features[i])):  # 把标签和特征列表都转换为numpy数组
            error += 1
    # 返回准确率
    return 1 - error / len(labels)


# *****************
# 运行
# *****************
if __name__ == '__main__':
    start = time()

    # 获取训练和测试数据集
    train_features, train_labels = load_data(r'./dataset/mnist_train.csv')
    test_features, test_labels = load_data(r'./dataset/mnist_test.csv')


    # 开始训练，学习w
    print('开始训练')
    w = logist_regression(train_features, train_labels)

    #验证准确率
    print('开始测试')
    accuracy = model_test(test_features, test_labels, w)
    end = time()
    # 打印准确率和运行时间
    print(f'准确率为：{accuracy}')
    print(f'运行时间为：{end-start}')
