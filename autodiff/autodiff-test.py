from collections import defaultdict
import numpy as np
class Variable:
    def __init__(self, value, local_gradients=[],name=''):
        self.value = value
        self.local_gradients = local_gradients
        self.name=name

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
    gradients = defaultdict(lambda: 0)
    def compute_gradients(variable, path_value):
        for child_variable, local_gradient in variable.local_gradients:
            value_of_path_to_child = path_value * local_gradient
            gradients[child_variable] += value_of_path_to_child
            print(child_variable.name+':',path_value,local_gradient,gradients[child_variable])
            compute_gradients(child_variable, value_of_path_to_child)

    compute_gradients(variable, path_value=1)
    return gradients

t = Variable(1,name='t')
x = Variable(2,name='v2')*t;x.name='x'
y = Variable(3,name='v3')*t;y.name='y'
d = Variable(4,name='v4')*x+Variable(5,name='v5')*y;d.name='d'

#t=1,x=2,y=3,d=8+15=23
gradients = get_gradients(d)
print('dy/dt=%f'% (gradients[t]))
