import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm

'''
K近邻算法手工复现， 使用mnist数据集
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
        features.append(item[1:])
    return features, labels  # features=[[28*28个元素],[],[]...] labels=[3, 6, 7, ...]


# *****************************
# 函数：计算两个特征向量之间的距离
# *****************************
def calculate_dist(x1, x2):
    # TODO:查一下向量之间都有哪些距离计算方式
    # 我们这里用的是欧式距离
    # 特征向量每个元素相减的平方，再求和，再开方
    return np.sqrt(np.sum(np.square(x1-x2)))


# *****************************************************
# 通过calculate_dist()找到与给定测试实例x距离最近的K个点
# 然后，计算K个点的类别频次，从确定x的类别
# *****************************************************
def get_closest(x, features, labels, top_K):
    # 建立一个列表，存放特征x与每个训练样本特征之间的距离
    dist_list = [0] * len(labels)
    for i, feat in enumerate(features):  # 遍历每个样本特征  [[], [], [],...]
        dist = calculate_dist(np.array(x), np.array(feat))
        dist_list[i] = dist
    # 得到dist_list中最小的K个元素所对应的位置
    top_index = np.argsort(np.array(dist_list))[:top_K]  # 升序排序，但排序的不是元素本身，而是其索引：[4,3,2,1] -> [3, 2, 1, 0]
    candidates = [labels[i] for i in top_index]  # candidates是邻居的标签：[4, 6, 7, 9, 0, ...]
    # 找出condidates中出现次数最多的元素
    res = max(candidates, key=candidates.count)
    return res


# ************************
# 定量测试模型正确率
# ************************
def model_test(train_features, train_labels, test_features, test_labels, top_K):
    print('start test')
    error_num = 0  # 初始化错误数量
    for i in tqdm(range(len(test_features))):
        x = test_features[i]
        y = get_closest(x, train_features, train_labels, top_K)
        if y != test_labels[i]: error_num += 1
    return 1- (error_num / len(test_features))




if __name__ == '__main__':
    start = time()

    # 获取训练数据
    train_features, train_labels = load_data(r'./dataset/mnist_train.csv')
    # 获取测试数据
    test_featrues, test_labels = load_data(r'./dataset/mnist_test.csv')
    # 计算准确率
    acc = model_test(train_features, train_labels, test_featrues, test_labels, top_K=30)
    print(f'模型的准确率为：{acc}')

    end = time()
    print(f'模型的运行时间为：{end-start}')

