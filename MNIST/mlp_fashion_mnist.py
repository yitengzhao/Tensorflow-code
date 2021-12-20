import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
x_train = x_train.reshape((-1,784))
x_test = x_test.reshape((-1,784))
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(784,)))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=512, verbose=1, epochs=100)
yp = model.predict(x_test)
print(model.evaluate(x_test,y_test))
