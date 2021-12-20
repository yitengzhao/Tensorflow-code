import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
x=np.arange(-6,6,0.1)
y=x**2+3
plt.plot(x,y)
plt.show()

x=tf.constant([6.,0.,150])
for step in range(500):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y=x**2+3
    grads=tape.gradient(y,[x])[0]
    x-=0.01*grads
    if step%20 == 19:
        print(f'step{step}: x ={x.numpy()},f(x)={y.numpy()}')