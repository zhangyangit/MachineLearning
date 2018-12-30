# -*- coding: utf-8 -*-
# 基类
__author__ = 'Morgan'
__date__ = '2018/12/17 23:42'

import numpy as np
# 导入 Cluster 类计算信息量
from c_CvDTree.Cluster import Cluster

# 定义抽象基类
class CvDNode:
    '''

    '''
    def __init__(self, tree=None, base=2, chaos=None, depth=0, parent=None, is_root=True, prev_feat="Root"):
        self._x = self._y = None
        self.base , self.chaos = base, chaos
        self.criterion = self.category = None
        self.left_child = self.right_child = None
        self._children, self.leafs = {}, {}
        self.sample_weight = None
        self.wc = None
        self.tree = tree

        # 如果传入Tree，进行初始化
        if tree is not None:
            # 由于数据预处理是由 Tree 完成
            self.wc = tree.whether_continuous
            # node 变量记录 Node 列表
            tree.nodes.append(self)

        self.feature_dim, self.tar, self.feats = None, None, {}
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feat
        self.is_cart = self.is_continuous = self.pruned = False

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    # 重载 __lt__ 方法
    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    # 重载 __str__ 和 __repr__ 方法
    def __str__(self):
        if self.category is None:
            return "CvDNode ({}) ({} -> {})".format(self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode ({}) ({} -> close; {})".format(self._depth, self.prev_feat, self.tree.label_dic[self.category])

    __repr__ = __str__

    # 定义 children 属性，主要是区分开连续+CART
    @property
    def children(self):
        return {
            "left": self.left_child, "right": self.right_child
        } if (self.is_cart or self.is_continuous) else self._children

    # 定义 height 属性
    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height if _child is not None else 0
                        for _child in self.children.values()])

    # 定义 info_dic 属性，记录Node主要信息
    @property
    def info_dic(self):
        return {"chaos": self.chaos, "y": self._y}


    # 定义第一种停止准则： 当特征维度为0或者Node的数据的不准确性小于阈值∈时停止
    # 同时，如果用户指定决策树的最大深度，那么当Node的深度太深时也停止
    # 若满足了停止条件, 该函数会返回 True，否则返回 False
    def stop1(self, eps):
        if(self._x.shape[1] == 0 or (self.chaos is not None and self.chaos <= eps) or (self.tree.max_depth is not None)):
            # 调用处理停止情况方法
            self._handle_terminate()
            return True
        return False

    # 定义第二种停止准则：当最大信息增益仍然小于阈值∈时停止
    def stop2(self, max_gain, eps):
        if max_gain <= eps:
            self._handle_terminate()
            return True
        return False

    # 利用 bincount 方法根据数据生成Node用所属类别的方法
    def get_category(self):
        return np.argmax(np.bincount(self._y))

    # 定义处理停止情况的方法，核心思想就是把Node
    def _handle_terminate(self):
        # 首先要生成 Node 所属类别
        self.category = self.get_category()
        # 然后一路回溯，更新父节点等，记录叶子节点 leafs
        _parent = self.parent
        while _parent is not None:
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent

    # 算法核心：
    #   1. 定义一个方法使其能将一个有子节点的Node转化为叶节点(局部剪枝)
    #   2. 定义一个方法使其能挑选出最好的划分标准
    #   3. 定义一个方法使其能根据划分标准进行生成

    # 局部剪枝
    def prune(self):
        # 调用相应方法进行计算 Node 所属类别
        self.category = self.get_category()
        # 记录由于Node转化为叶节点而被剪去的
        _pop_lst = [key for key in self.leafs]
        # 然后一路回溯，更新各个parent属性leafs
        _parent = self.parent
        while _parent is not None:
            for _k in _pop_lst:
                # 删去局部剪枝而被减掉的叶节点
                _parent.leafs.pop(_k)
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent
        # 调用mark_pround方法将自己所有子节点的 pround 属性置 True
        self.mark_pruned()
        self.feature_dim = None
        self.left_child = self.right_child = None
        self._children = {}
        self.leafs = {}

    # 定义给局部剪枝减掉的Node打标记
    def mark_pruned(self):
        self.pruned = True
        # 遍历各个子节点
        for _child in self.children.values():
            # 如果当前子节点不是Node，递归调用mark_pruned方法
            # (连续型特征和CART算法， 可能导致 children 中出现None
            if _child is not None:
                _child.mark_pruned()

    # 挑选最佳划分标准的方法，分清二分，多分
    def fit(self, x, y, sample_weight, eps=1e-8):
        self._x, self._y = np.atleast_2d(x), np.array(y)
        self.sample_weight = sample_weight
        # 满足第一种停止准则，退出函数体
        if self.stop1(eps):
            return
        # 用该 Node 数据实例化 Cluster 计算各种信息量
        _cluster = Cluster(self._x, self._y, sample_weight, self.base)
        # 对于根节点，需要额外计算数据的不确定性
        if self.is_root:
            if self.criterion == "gini":
                self.chaos = _cluster.gini()
            else:
                self.chaos = _cluster.ent()
        _max_gain, _chaos_lst = 0, []
        _max_feature = _max_tar = None
        # 遍历还能选择的特征
        for feat in self.feats:
            # 如果连续型或是CART算法，需要额外计算二分标准的取值集合
            if self.wc[feat]:
                _samples = np.sort(self._x.T[feat])
                _set = (_samples[:-1] + _samples[1:]) * 0.5
            elif self.is_cart:
                _set = self.tree.feature_sets[feat]
            # 遍历二分标准并调用二类问题相关的计算信息量的方法
            if self.is_cart or self.wc[feat]:
                for tar in _set:
                    _tmp_gain, _tmp_chaos_lst = _cluster.b