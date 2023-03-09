import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm

'''
朴素贝叶斯算法手工复现， 使用mnist数据集
mnist手写体识别数据集格式：[[3, 28*28个元素], [], []]
'''

# *********************
# 函数：加载mnist数据集
# *********************
def load_data(file_path):
    print('开始加载数据')
    features = []  # 存放样本特征的列表
    labels = []  # 存放样本标签的列表
    data = pd.read_csv(file_path)  # 此时data是pandas.DataFrame类型的数据，类似于excel表格，有表头和行号
    data_list = data.values.tolist()  # DataFrame -> 二维列表
    for item in data_list:  # 循环读取每一个样本实例
        labels.append(item[0])
        features.append([0 if F<128 else 1 for F in item[1:]])  # 对特征进行离散化处理
    return features, labels  # features=[[28*28个元素],[],[]...] labels=[3, 6, 7, ...]


# ********************************
# 计算先验概率Py，计算条件概率Px_y
# ********************************
def get_probability(features, labels):
    # 计算每个标签出现的概率（先验概率）
    num_category = 10  # 类别数，共10个类别
    num_features = 28*28
    num_data = len(labels)  # 样本数，共10000个样本
    Py = np.zeros((num_category, 1))  # 初始化10个类别的先验概率
    for i in range(num_category):
        Py[i] = ((np.sum(np.array(labels) == i)) + 1) / (num_data + 10)  # 计算每个类别的先验概率
    Py = np.log(Py)

    # 计算条件概率
    Px_y = np.zeros((num_category, num_features, 2))  # [[[], [0.8, 0.2], []], [], []]
    for i in tqdm(range(num_data)):
        label = labels[i]
        x = features[i]  # x=[28*28个元素]
        for j in range(num_features):
            Px_y[label][j][x[j]] += 1

    for label in tqdm(range(num_category)):  # 循环每一个类别对应的每一个特征
        for j in range(num_features):
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            Px_y[label][j][0] = np.log((Px_y0+1) / (Px_y0+Px_y1+2))
            Px_y[label][j][1] = np.log((Px_y1+1) / (Px_y0+Px_y1+2))

    return Py, Px_y

# ****************************************************
# 朴素贝叶斯概率估计函数
# Py是先验概率，Px_y是条件概率，x是待测样本特征
# ****************************************************
def naivebayes(Py, Px_y, x):
    # 设置特征维度
    feature_dim = 28*28
    # 设置类别数目
    num_labels = 10
    # 建立存放所有标签的估计概率数组
    P = [0] * num_labels
    for i in range(num_labels):  # 对于每一个类别，单独计算其概率
        # 初始化sum为0，sum为求和项
        # 在训练过程中对概率进行了log处理，使累乘求积变为累加求和
        sum = 0
        for j in range(feature_dim):  # 获取每一个条件概率值，进行累加
            sum += Px_y[i][j][x[j]]
        # 最后再和先验概率相加
        P[i] = sum + Py[i]

    return P.index(max(P))


# *********************
# 模型测试函数
# *********************
def model_test(Py, Px_y, features, labels):
    # 错误计数值
    error = 0
    for i in tqdm(range(len(labels))):  # 循环遍历测试集中的每一个样本
        # 获取预测值
        pred = naivebayes(Py, Px_y, features[i])
        if pred != labels[i]:
            error += 1
    return 1 - (error / len(labels))


# **********************
# 运行
# **********************
if __name__ == '__main__':
    start = time()

    # 获取训练集和测试集
    train_features, train_labels = load_data(r'./dataset/mnist_train.csv')
    test_features, test_labels = load_data(r'./dataset/mnist_test.csv')

    #开始训练，学习先验分布和条件概率分布
    Py, Px_y = get_probability(train_features, train_labels)
    accuracy = model_test(Py, Px_y, test_features, test_labels)
    end = time()
    # 打印准确率和运行时间
    print(f'准确率为：{accuracy}')
    print(f'运行时间为：{end-start}')

