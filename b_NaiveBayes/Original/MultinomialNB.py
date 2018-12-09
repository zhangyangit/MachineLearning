# -*- coding: utf-8 -*-
# 导入基本架构 Basic

from b_NaiveBayes.Original.Basic import *

class MultinomialNB(NaiveBayes):
    # 定义预处理数据方法
    def feed_data(self, x, y, sample_weight=None):
        if isinstance(x, list):
            features = map(list, zip(*x))
        else:
            features = x.T

        # 利用python的集合处理数据,利用 bincount 优化算法，从特征 0 开始数值化
        # 注意：数值化过程中的转换关系记录成字典，否则无法对数据进行判断
        features = [set(feat) for feat in features]
        feat_dics = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        label_dic = {_l: i for i, _l, in enumerate(set(y))}

        # 利用转换字典更新训练集
        x = np.array([[feat_dics[i][_l] for i, _l in enumerate(sample)] for sample in x])
        y = np.array([label_dic[yy] for yy in y])

        # 利用 bincount 获取类别数据
        cat_counter = np.bincount(y)

        # 记录维度特征取值数
        n_possibilites = [len(feats) for feats in features]
        # 获取类别数据下标
        labels = [y == value for value in range(len(cat_counter))]
        # 利用下标获取记录并记录类别分开后的输入数据的数组
        labelled_x = [x[ci].T for ci in labels]
        # 更新模型的各个属性
        self._x, self._y = x, y
        self._labelled_x, self._labelled_zip = labelled_x, list(zip(labels, labelled_x))
        (self._cat_counter, self._feat_dics, self._n_possibilities) = (cat_counter, feat_dics, n_possibilites)
        self.label_dic = {i: _l for _l, i in label_dic.items()}
        # 调用处理权重的函数，更新记录条件概率的数组
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        # 利用Numpy bincount 获取带权重的条件概率极大似然估计
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([
                    np.bincount(xx[dim], minlength=_p) for xx in self._labelled_x])
            else:
                local_weights = sample_weight * len(sample_weight)
                self._con_counter.append([
                    np.bincount(xx[dim], weights=local_weights[label], minlength=_p)
                    for label, xx in self._labelled_zip])

    # 定义核心训练函数
    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        # data 存储之后加了平滑项后的条件概率的数组
        data = [[] for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [
                [(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities)
                for p in range(n_possibilities)] for c in range(n_category)]
        self._data = [np.asarray(dim_info) for dim_info in data]

        # 利用data 函数生成决策函数
        def func(input_x, tar_category):
            rs = 1
            # 遍历各个维度， 利用data 和条件独立性假设计算联合条件概率
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category][xx]
            return rs * p_category[tar_category]
        # 返回决策函数
        return func

    def _transfer_x(self, x):
        # 遍历元素
        for j, char in enumerate(x):
            x[j] = self._feat_dics[j][char]
        return x

if __name__=='__main__':
    # 导入 标准库，读取数据
    import time
    from Util.Util import DataUtil
    # 遍历数据集
    #dataset = "mushroom"
    for dataset in ("balloon1.0", "balloon1.5"):
        # 读入数据TypeError: 'NoneType' object is not subscriptable
        _x, _y = DataUtil.get_dataset(dataset, "../../data/{}.txt".format(dataset))
        # 实例化模型，训练，记录时间
        learning_time = time.time()
        nb = MultinomialNB()
        nb.fit(_x, _y)
        learning_time = time.time() - learning_time
        # 评估模型表现，记录话费时间
        estimation_time = time.time()
        nb.evaluate(_x, _y)
        estimation_time = time.time() - estimation_time
        # 打印
        print(
            "Model Building : {:12.6} s\n"
            "Estimation     : {:12.6} s\n"
            "Total          : {:12.6} s\n".format(learning_time, estimation_time, learning_time+estimation_time)
        )