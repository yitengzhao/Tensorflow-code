import numpy as np
x=np.array(range(10)).reshape((10,1))
y=x*x+1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(100,activation='sigmoid',input_shape=(1,)))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=10000)
yp = model.predict(x)
import matplotlib.pyplot as plt
plt.plot(x,y,'.-',x,yp)
plt.show()
