import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,1,1,0])
model=Sequential()
model.add(Dense(2,activation='sigmoid',input_shape=(2,)))
model.add(Dense(1))
model.compile(loss='mse',optimizer=tensorflow.keras.optimizers.SGD(lr=0.1))
history = model.fit(x,y, epochs=10000)
print(model.predict(x))
