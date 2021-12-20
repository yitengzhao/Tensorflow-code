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
    return ONE / (ONE + exp(-z))

x = Variable(1)
w = Variable(0)
b = Variable(0)
Y = Variable(1)
y = sigmoid(w * x + b)
gradients = get_gradients(y)
print('dy/dw=%f, dy/db=%f' % (gradients[w], gradients[b]))
for i in range(100):
    y = sigmoid(w * x + b)
    C = (y - Y) * (y - Y)
    print('Cost=%f,y=%f' % (C.value, y.value))
    gradients = get_gradients(C)
    w.value = w.value - 0.1 * gradients[w]
    b.value = b.value - 0.1 * gradients[b]
