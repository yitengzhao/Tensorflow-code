import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32,3,activation='relu',padding='same',input_shape=((28,28,1))))
model.add(MaxPooling2D(2))
model.add(Conv2D(64,3,activation='relu',padding='same'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=512, verbose=1, epochs=20)
yp = model.predict(x_test)
print(model.evaluate(x_test,y_test))
