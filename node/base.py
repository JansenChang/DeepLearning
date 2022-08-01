import numpy as np


def sigmoid(x):  # sigmoid 函数
    return 1 / (1 + np.exp(-x))


def step_function(x):  # 阶跃函数
    return np.array(x > 0, int)


def softmax(a):  # softmax函数
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出策略
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def mean_squared_error(y: np.array, t: np.array) -> object:  # 均方误差函数
    """

    :rtype: float
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y: np.array, t: np.array) -> object:  # 交叉熵误差
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))
