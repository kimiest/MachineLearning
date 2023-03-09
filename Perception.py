import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm

'''
感知机算法手工复现， 使用mnist数据集
mnist手写体识别数据集格式：[[3, 28*28个int], [], []]
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
        L = -1 if item[0] < 5 else 1   # 由于感知机是用于二分类任务的，需要将10个标签映射到[-1,1]两个标签上
        labels.append(L)
        features.append([int(F)/255 for F in item[1:]])  # 标准化
    return features, labels  # features=[[28*28个元素],[],[]...] labels=[-1, 1, 1, ...]

# *******************
# 感知机函数
# *******************
def preception(features, labels, total_epoch=50, lr=0.0001):
    print('开始训练')
    feature_dim = len(features[0])  # 特征维度
    num_data = len(labels)
    w = np.zeros((1, feature_dim))  # 创建初始化权重w，初始值为0，与特征向量维度保持一致
    b = 0  # 创建初始化偏置b，初始值为0

    # 拿训练数据来更新w和b
    bar = tqdm(range(total_epoch))
    for epoch in bar:
        # 记录一个epoch内的总损失
        epoch_loss = 0.0
        # 拿到一个训练数据，进行一次更新
        for i in range(num_data):
            xi = np.array(features[i])  # xi是一个28*28维的向量，即第i个样本的特征向量
            yi = np.array(labels[i])  # 第i个样本的标签
            loss = -1 * yi * (np.dot(w, xi) + b)  # 计算当前样本的损失值
            if  loss >= 0:  # 如果损失值大于/等于0，则说明该样本预测错误，那么调整w和b
                # 梯度下降修改w和b
                w = w + lr * yi * xi
                b = b + lr * yi
                epoch_loss += loss
        bar.set_postfix(loss=epoch_loss/num_data)  # 在tqdm中展示每个epoch的平均损失
    return w, b

# ****************
# 模型测试函数
# ****************
def model_test(features, labels, w, b):
    print('开始测试')
    error = 0
    # 遍历所有测试样本
    for i in tqdm(range(len(labels))):
        xi = features[i]  # 第i个测试样本的特征向量
        yi = labels[i]  # 第i个测试样本的标签
        res = -1 * yi * (np.dot(w, xi) + b)  # 输出预测结果
        if res >= 0: error += 1  # 如果预测错误，则error+1
    # 计算最终的准确率
    acc = 1 - (error / len(labels))
    return acc


if __name__ == '__main__':
    # 获取开始时间
    start = time()

    # 获取训练/验证数据的特征及标签
    train_features, train_labels = load_data(r'.\dataset\mnist_train.csv')
    test_features, test_labels = load_data(r'.\dataset\mnist_test.csv')

    # 训练权重w和偏置b
    w, b = preception(train_features, train_labels)
    acc = model_test(test_features, test_labels, w, b)

    # 获取结束时间
    end = time()

    # 打印准确率和运行时间
    print(f'准确率为{acc}，运行时间为：{end-start}')




