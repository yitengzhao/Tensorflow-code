import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def himmelblau(x): # himmelblau 
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y) # generate x-y points
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y]) # compute Z
fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
x = tf.constant([4., 0. ]) # init
for step in range(200):# loop 200 times
    with tf.GradientTape() as tape: # gradient
        tape.watch([x]) # add to gradient list
        y = himmelblau(x) # feedforward
    grads = tape.gradient(y, [x])[0] 
    x -= 0.01*grads  # lr=0.01
    if step % 20 == 19: # print min
        print ('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))
