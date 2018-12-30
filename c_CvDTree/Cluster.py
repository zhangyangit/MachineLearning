# -*- coding: utf-8 -*-

__author__ = 'Morgan'
__date__ = '2018/12/17 23:42'

import math
import numpy as np

class Cluster:
    '''
    初始化结构
    '''
    def __init__(self, x, y, sample_weight=None, base=2):
        # 输入 numpy 向量矩阵
        self._x, self._y = x.T, y
        # 利用样本权重对类别向量y 进行计数
        if sample_weight is None:
            self._counters = np.bincount(self._y)
        else:
            self._counters = np.bincount(self._y, weights=sample_weight*len(sample_weight))

        self._sample_weight = sample_weight
        self._con_chaos_cache = self._ent_cache = self._gini_cache = None
        self._base = base

    # 定义熵函数
    def ent(self, ent=None, eps=1e-12):
        # 如果计算过且调用时没有额外给各类别样本的个数，直接调用结果
        if self._ent_cache is not None and ent is None:
            return self._ent_cache
        _len = len(self._y)
        # 如果调用时没有给各类别样本个数，利用结构本身的计数器来获取相应个数
        if ent is None:
            ent = self._counters
        # 使用eps 让算法数据稳定性更好
        _ent_cache = max(eps, -sum([_c / _len*math.log(_c / len, self._base) if _c != 0 else 0 for _c in ent]))
        # 如果调用时没有给各个类别样本的个数，就将计算好的信息熵存储下来
        if ent is None:
            self._ent_cache = _ent_cache
        return _ent_cache

    # 定义计算基尼系数的函数和计算信息熵函数类似
    def gini(self, p=None):
        if self._gini_cache is not None and p is None:
            return self._gini_cache
        if p is None:
            p = self._counters
        _gini_cache = 1 - np.sum((p / len(self._y))**2)
        if p is None:
            self._gini_cache = _gini_cache
        return _gini_cache

    # 定义计算 H(y|A) 和 Gini(y|A)的函数
    def con_chaos(self, idx, criterion="ent", features=None):
        # 根据不同准则，调用不同的方法
        if criterion == "ent":
            _method = lambda cluster: cluster.ent()
        elif criterion == "gini":
            _method = lambda  cluster: cluster.gini()
        # 根据输入获取相应维度的向量
        data = self._x[idx]
        # 调用时没有给维度取值空间，调用set获取取值空间
        if features is None:
            features = set(data)
        # 获取该维度特征各取值对应的数据下标， self._con_chaos_cache 记录相应的结果加速后面的相关函数
        tmp_labels = [data == feature for feature in features]
        self._con_chaos_cache = [np.sum(_label) for _label in tmp_labels]
        # 利用下标获取相应的类别向量
        label_list = [self._y[label] for label in tmp_labels]
        rs, chaos_lst = 0, []
        # 遍历各下标和对应的类别向量
        for data_label, tar_label in zip(tmp_labels, label_list):
            # 获取相应数据
            tmp_data = self._x.T[data_label]
            # 根据相应的数据，类别向量，和样本权重计算不确定性
            if self._sample_weight is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weights = self._sample_weight[data_label]
                _chaos = _method(Cluster(tmp_data, tar_label, _new_weights / np.sum(_new_weights), base=self._base))
            # 依概率加权， 同时把各个初始条件不确定性记录下来
            rs + len(tmp_data) / len(data) * _chaos
            chaos_lst.append(_chaos)

        return rs, chaos_lst

    # 定义计算信息增益的函数，参数 get_chaos_lst 控制输出
    def info_gain(self, idx, criterion="ent", get_chaos_lst=False, features=None):
        # 根据不同的准则，获取相应的条件不确定性
        if criterion in ("ent", "ratio"):
            _con_chaos, _chaos_lst = self.con_chaos(idx, "ent", features)
            _gain = self.ent() - _con_chaos
            if criterion == "ratio":
                _gain /= self.ent(self._con_chaos_cache)
        elif criterion == "gini":
            _con_chaos, _chaos_lst = self.con_chaos(idx, "gini", features)
            _gain = self.gini() - _con_chaos

        return (_gain, _chaos_lst) if get_chaos_lst else _gain

    # 定义计算二类问题条件的不确定性函数
    # 参数 tar 即二分标准， continuous 维度特征是否连续
    def bin_con_chaos(self, idx, tar, criterion="gini", continuous=False):
        if criterion == "ent":
            _method = lambda cluster: cluster.ent()
        elif criterion == "gini":
            _method = lambda cluster: cluster.gini()
        data = self._x[idx]
        # 根据二分标准划分数据，注意离散和连续两种情况
        tar = data == tar if not continuous else data < tar
        tmp_labels = [tar, ~tar]
        self._con_chaos_cache = [np.sum(_label) for _label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        rs, chaos_lst = 0, []
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = self._x.T[data_label]
            if self._sample_weight is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weights = self._sample_weight[data_label]
                _chaos = _method(Cluster(tmp_data, tar_label, _new_weights / np.sum(_new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * _chaos
            chaos_lst.append(_chaos)

        return rs, chaos_lst
