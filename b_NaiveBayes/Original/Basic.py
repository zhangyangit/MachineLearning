# -*- coding: utf-8 -*-
# 朴素贝叶斯算法
# input: 训练数据集 D = {(x1, y1),````,(xN, yN) }
# course: 利用 ML 估计导出模型参数
#          1. 计算先验概率 p(y=ck)的极大似然估计
#          2. 计算条件概率 p(x(j)=ajl | y=ck) 的极大似然估计
# output: 利用 MAP 估计决策
# 思想：
# 1. 使用 ML 估计导出模型参数  (先验概率，条件概率)
# 2. 使用 MAP 估计作为模型的决策 (输出使得数据 MAP 最大化)

# 导入使用的库
import numpy as np

# 定义基类
class NaiveBayes:
    '''

    '''
    def __init__(self):
        self._x = self._y = None
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._labelled_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dic = self._feat_dics = None

    # 重载
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    # 抽象方法
    def feed_data(self, x, y, sample_weight=None):
        pass

    # 抽象方法
    def feed_sample_weight(self, sample_weight=None):
        pass

    # 定义计算先验概率函数胡 lb 是估计项的平滑项，默认1
    def get_prior_probability(self, lb=1):
        return [(_c_num + lb) / (len(self._y) + lb * len(self._cat_counter))
                for _c_num in self._cat_counter]

    # 定义普适性的训练函数
    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        # 传入 x, y
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        # 调用核心函数

        self._func = self._fit(lb)

    # 定义抽象算法子类定义
    def _fit(self, lb):
        pass

    # 定义单一预测函数
    def predict_one(self, x, get_raw_result=False):
        # 预测前 ，数据数值化
        if isinstance(x, np.ndarray):
            x = x.tolist()
        # 数据拷贝
        else:
            x = x[:]
        # 调用 方法数值化
        x = self._transfer_x(x)
        m_arg, m_probability = 0, 0

        # 遍历
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p

            if not get_raw_result:
                return self.label_dic[m_arg]
            return m_probability

    # 定义多样本预测函数
    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])

    # 定义对数据评估方法
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        print("Acc: {:12.6} %".format(100 * np.sum(y_pred == y) / len(y)))

    # 定义数值化函数
    def _transfer_x(self, x):
        # 遍历元素
        return x
