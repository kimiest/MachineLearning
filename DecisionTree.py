import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
from collections import defaultdict
'''
决策树算法手工复现， 使用mnist数据集
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
        features.append([0 if F<128 else 1 for F in item[1:]])  # 将特征二值离散化
    return features, labels  # features=[[28*28个元素],[],[]...] labels=[3, 6, 7, ...]


# ******************************
# 找到当前标签集中占数目最大的标签
# ******************************
def major_category(labels):
    count = defaultdict(int)
    # 遍历所有标签
    for i, L in enumerate(labels):
        count[L] += 1  # 标签L的数量+1
    # 对字典依据value进行排序
    freq_sorted = sorted(count.items(), key=lambda x:x[1], reverse=True)
    # 返回最大项的所对应的标签
    return freq_sorted[0][0]


# **********************
# 计算经验熵
# **********************
def cal_H_D(labels):
    # 首先初始化经验熵为0
    H_D = 0
    unique = set(labels)
    for L in unique:  # 遍历每一个出现过的标签
        P = labels[labels == i].size / labels.size
        # 对经验熵的每一项累加求和
        H_D += -1 * P * np.log2(P)

    return H_D

# **************************
# 计算条件熵
# *************************
def cal_H_D_A(features, labels):
    # 初始为0
    H_D_A = 0
    feat_set = set([L for L in features])

    # 对于每一个特征取值遍历计算条件熵的每一项
    for F in feat_set:
        H_D_A += features[features == i].size / fetures.size * cal_H_D(labels[features == i])
    return H_D_A

# *************************
# 计算信息增益最大的特征
# *************************
def calc_best_features(features, labels):
    # 将特征和标签列表转换为numpy数组
    features = np.array(features)
    labels = np.array(labels)

    # 获取当前特征数目
    num_feat = features.shape[1]

    # 初始化最大信息增益
    maxG_D_A = -1
    # 初始化最大信息增益的特征
    max_feat = -1
    # 计算经验熵
    H_D = cal_H_D(labels)
    for n in num_feat:  # 对每一个特征进行遍历计算
        features_divided = np.array(features[:, n].flat)
        G_D_A = H_D - cal_H_D_A(features, labels)
        if G_D_A > maxG_D_A:
            maxG_D_A = G_D_A
            max_feat = n
    return max_feat, maxG_D_A


# **************************
# 更新特征集和标签集
# **************************
def get_sub_features(features, labels, a, A):
    # 返回的特征集
    return_features = []
    # 返回的标签集
    return_labels = []
    # 对当前数据的每一个样本进行遍历
    for i in range(len(labels)):
        # 如果当前样本的特征为指定特征值a
        if features[i][A] == a:
            # 那么将该样本的第A个特征切割掉，放入返回的特征集中
            return_features.append(features[i][0:A] + features[i][A+1:])
            # 将该样本的标签放入返回标签中
            return_labels.append(labels[i])
    return return_features, return_labels


# ************************
# 创建决策树
# ************************
def create_tree(features, labels):
    Epsilo = 0.1
    feat = features[0]
    L = labels[0]
    print(f'开始一个节点，{len(feat[0])}，{len(L[0])}')

    classdict = (i for i in feat)
    if len(classdict) == 1:
        return feat[0]

    if len(feat[0]) == 0:
        return major_category(feat)

    Ag, Epsiloget = calc_best_features(feat, L)

    if Epsiloget < Epsilo:
        return major_category(L)

    treedict = {Ag:{}}
    # 特征为0时，进入0分支
    treedict[Ag][0] = create_tree(get_sub_features(features, labels, Ag, 0))
    treedict[Ag][1] = create_tree(get_sub_features(features, labels, Ag, 1))
    return treedict


# *************************
# 预测
# *************************
def predict(features, tree):
    # 一个循环，一直找，直到找到一个有效的分类    {a:{d: {}}, f: 44}, b:45, c:56}
    while True:
        key, value = tree.items()
        if type(tree[key].__name__ == 'dict'):
            data_val = features[key]
            del features[key]
            tree = value[data_val]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value

# *****************
# 模型测试函数
# *****************
def model_test(features, labels, tree):
    # 错误计数
    error = 0
    for i in range(len(labels)):
        if labels[i] != predict(features, labels):
            error += 1
    return 1 - error/len(labels)

# *******************
# 运行
# *******************
if __name__ == '__main__':
    start = time()

    # 获取训练和测试数据集
    train_features, train_labels = load_data(r'./dataset/mnist_train.csv')
    test_features, test_labels = load_data(r'./dataset/mnist_test.csv')
    tree = create_tree(train_features, train_labels)
    accuracy = model_test(test_features, test_labels, tree)
    end = time()
    print(f'准确率为{accuracy}')
    print(f'运行时间为{end-start}')




