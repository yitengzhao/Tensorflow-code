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

w = [Variable(np.random.randn()) for i in range(9)]
def f(x1,x2):
    x1 = Variable(x1)
    x2 = Variable(x2)
    h1 = sigmoid(x1*w[0]+x2*w[1]+w[2])
    h2 =sigmoid(x1*w[3]+x2*w[4]+w[5])
    ho =sigmoid(w[6]*h1+w[7]*h2+w[8])
    return ho

for i in range(20000):
    C = (f(0, 0) - Variable(0)) * (f(0, 0) - Variable(0))
    C = (f(0, 1) - Variable(1)) * (f(0, 1) - Variable(1)) + C
    C = (f(1, 0) - Variable(1)) * (f(1, 0) - Variable(1)) + C
    C = (f(1, 1) - Variable(0)) * (f(1, 1) - Variable(0)) + C
    print('Epoch %d, Cost=%f' % (i,C.value))
    gradients = get_gradients(C)
    for j in range(len(w)):
        w[j].value = w[j].value - 0.1 * gradients[w[j]]

print("f(0,0)=%f" % f(0,0).value)
print("f(0,1)=%f" % f(0,1).value)
print("f(1,0)=%f" % f(1,0).value)
print("f(1,1)=%f" % f(1,1).value)
