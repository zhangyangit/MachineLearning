# -*- coding: utf-8 -*-

__author__ = 'Morgan'
__date__ = '2018/9/16 14:04'

# 导入需要用到的库
import numpy as np
import matplotlib.pyplot as plt

# 获取并处理数据
# 定义存储输入数据(x) 和 目标数据(y) 的数组
x, y = [], []
# 遍历数据集， 变量sample 对应的正是一个个样本
for sample in open("../data/prices.txt", "r"):
    # 由于数据是用逗号隔开的，所以调用python 中的 split 方法并将逗号作为参数传入
    _x, _y = sample.split(',')
    # 将字符串数据转化为浮点数
    x.append(float(_x))
    y.append(float(_y))

# 读取完数据后，将他们转化为Numpy 数组，以方便进一步处理
x, y = np.array(x), np.array(y)
# 标准化
x = (x - x.mean()) / x.std()
# 将原始数据以散点图的形式的画出
plt.figure()
plt.scatter(x, y, c="g", s=6)
plt.savefig("../output/scatter_diagram.jpg")
plt.show()

# 选择和训练模型
# 在(-2, 4) 这个区间上取 100 个点做画图的基础
x0 = np.linspace(-2, 4, 100)
# 利用 Numpy 的函数定义训练并返回多项式回归模型的函数
# deg 参数代表着模型参数的n, 亦或模型中多项式的次数
# 返回模型能够根据输入的x (默认x0)， 返回相对应的预测的y
# polyfit(x, y, deg):
# 函数返回 L(p,n) = 1/2{[f(x|p;n)-y]^2} 积分的最小参数 p ,次函数是模型的训练函数
# polyval(p, x):
# 根据多项式
def get_model(deg):
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)

# 评估和可视化结果
# 根据参数 n, 输入的 x, y 返回相对应的损失
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()

# 定义测试参数集并根据它进行各种实验
test_set = (1, 4, 10)
for d in test_set:
    # 输出相应的损失
    print(get_cost(d, x, y))

# 画出相应的图像
plt.scatter(x, y, c="g", s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label="degree = {}".format(d))

# 将横轴，纵轴的范围分别限制在(-2, 4), (10^5, 8*10^5)
plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)
# 调用legend 方法使曲线对应的 label 正确显示
plt.legend()
plt.savefig("../output/LinearFitting_diagram.jpg")
plt.show()