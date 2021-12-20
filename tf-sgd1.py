import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
x = np.arange(-6, 6, 0.1)
y = x**2 + 3
plt.plot(x,y)
plt.show()
x = tf.constant([6., 0.]) # init
for step in range(200):# loop 200 times
    with tf.GradientTape() as tape: # gradient
        tape.watch([x]) # add to gradient list
        y = x**2+3 # feedforward
    grads = tape.gradient(y, [x])[0] 
    x -= 0.01*grads  # lr=0.01
    if step % 20 == 19: # print min
        print ('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))
