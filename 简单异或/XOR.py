import numpy as np
def step(x):
    return (np.sign(x)+1)/2
def f1(x,y):
    return step(x+y-0.5)
def f2(x,y):
    return step(-x-y+1.5)
def xor(x,y):
    return step(f1(x,y)+f2(x,y)-1.5)

x = np.array([0,0,1,1])
y = np.array([0,1,0,1])
z = xor(x,y)
for i in range(len(x)):
    print(f"xor({x[i]},{y[i]})={z[i]}")