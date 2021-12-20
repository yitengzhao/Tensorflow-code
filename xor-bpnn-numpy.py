import numpy as np


def act(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


NH = 10
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
w1 = 2 * np.random.random((2, NH)) - 1
w2 = 2 * np.random.random((NH, 1)) - 1


def feedfoward(x):
    a0 = x;
    a1 = act(np.dot(a0, w1))
    a2 = act(np.dot(a1, w2))
    return (a0, a1, a2)


n_epochs = 1000000
for i in range(n_epochs):
    a0, a1, a2 = feedfoward(x)
    l2_delta = (a2 - y) * act(a2, deriv=True)
    l1_delta = l2_delta.dot(w2.T) * act(a1, deriv=True)
    w2 = w2 - a1.T.dot(l2_delta) * 0.1
    w1 = w1 - a0.T.dot(l1_delta) * 0.1
    if (i % 10000) == 0:
        loss = np.mean(np.abs(y - a2))
        print("epochs %d/%d loss = %f" % (i / 1e4 + 1, n_epochs / 1e4, loss))

a0, a1, a2 = feedfoward(x)
print("xor(", x, ") = ", a2)