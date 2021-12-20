from collections import defaultdict
import numpy as np


class Variable:
    def __init__(self, value, local_gradients=[]):
        self.value = value
        self.local_gradients = local_gradients

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __sub__(self, other):
        return add(self, neg(other))

    def __neg__(self):
        return neg(self)

    def __truediv__(self, other):
        return mul(self, inv(other))

def add(a, b):
    value = a.value + b.value
    local_gradients = ((a, 1), (b, 1))
    return Variable(value, local_gradients)

def mul(a, b):
    value = a.value * b.value
    local_gradients = ((a, b.value), (b, a.value))
    return Variable(value, local_gradients)

def neg(a):
    value = -1 * a.value
    local_gradients = ((a, -1),)
    return Variable(value, local_gradients)

def inv(a):
    value = 1. / a.value
    local_gradients = ((a, -1 / a.value ** 2),)
    return Variable(value, local_gradients)

def exp(a):
    value = np.exp(a.value)
    local_gradients = ((a, value),)
    return Variable(value, local_gradients)


def get_gradients(variable):
    gradients = defaultdict(lambda: 0)  # 可以根据Variable变量地址索引对应的梯度
    def compute_gradients(variable, path_value):
        for child_variable, local_gradient in variable.local_gradients:  # 两条路径循环2次
            value_of_path_to_child = path_value * local_gradient  # 从后往前，乘以每条边的梯度
            gradients[child_variable] += value_of_path_to_child  # 不同路径的梯度相加，算的是局部偏微分
            compute_gradients(child_variable, value_of_path_to_child)  # 递归整个计算图

    compute_gradients(variable, path_value=1)  # path_value=1，输出对自己的偏微分为1
    return gradients

def sigmoid(z):
    ONE = Variable(1)
    return ONE/(ONE + exp(-z))

w1 = Variable(1)
x = Variable(1)
Y = Variable(1)
y = w1*x
for i in range(4):
    w = Variable(1)
    y = w*sigmoid(w*y)
C = (Y-y)*(Y-y)
gradients = get_gradients(C)
print('dC/dw1=%f' % gradients[w1])
