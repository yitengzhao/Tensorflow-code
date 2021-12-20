from collections import defaultdict
class Variable:
    def __init__(self, value, local_gradients=[]):
        self.value = value
        self.local_gradients = local_gradients
    def __add__(self, other):
        return add(self, other)

def add(a, b):
    value = a.value + b.value
    local_gradients = ((a, 1),(b, 1))
    return Variable(value, local_gradients)

def get_gradients(variable):
    gradients = defaultdict(lambda: 0)
    def compute_gradients(variable, path_value):
        for child_variable, local_gradient in variable.local_gradients:
            value_of_path_to_child = path_value * local_gradient
            gradients[child_variable] += value_of_path_to_child
            compute_gradients(child_variable, value_of_path_to_child)

    compute_gradients(variable, path_value=1)
    return gradients

x = Variable(1)
b = Variable(0)
y = x+b+b
gradients = get_gradients(y)
print('dy/dx=%f, dy/db=%f'% (gradients[x], gradients[b]))
