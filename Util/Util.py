# -*- coding: utf-8 -*-

import numpy as np


class DataUtil:
    # 定义方法从文件中读取数据
    # 该方法接收5个参数：
    #    数据集名字， 数据集路径， 训练样本数，类别所在列，是否打乱数据
    def get_dataset(name, path, train_num=None, tar_idx=None, shuffle=True):
        x = []
        print(name)
        print(path)
        # 将编码设为 utf-8
        with open(path, "r", encoding="utf-8") as file:
            # 如果是气球数据，直接根据逗号分割
            if "balloon" in name:
                for sample in file:
                    x.append(sample.strip().split(","))
        # 默认打乱数据
        if shuffle:
            np.random.shuffle(x)

        # 默认类别在最后一列
        tar_idx = -1 if tar_idx is None else tar_idx
        y = np.array([xx.pop(tar_idx) for xx in x])
        x = np.array(x)
        # 默认全部都是训练集
        if train_num is None:
            return x, y
        # 若传输训练样本数，则分割数据集
        return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])

