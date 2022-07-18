import numpy as np


def sigmoid(x):  # sigmoid 函数
    return 1 / (1 + np.exp(-x))


def step_function(x):  # 阶跃函数
    return np.array(x > 0, int)


def softmax(a): # softmax函数
    c = np.max(a)
    exp_a = np.exp(a-c) # 溢出策略
    sum_exp_a= np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
