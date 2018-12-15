import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import parse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

random.seed()

CATEGORIES = ["circle","triangle","star"]

#0 = circle 1 = triangle 2 = star

training = []

parse.get_traindata(training,CATEGORIES)

random.shuffle(training)


x = []
y = []

for i,j in training:
  x.append(i)
  y.append(j)


# for i in range(0,8):
#   plt.imshow(x[random.randint(0,len(x))],cmap='gray')
#   plt.show()

x = np.array(x).reshape(-1,50,50,1)
x = x/255.0

model = Sequential()

model.add(Conv2D(128,(3,3),input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='SGD', metrics=['accuracy'])

model.fit(x,y,batch_size=32,epochs=5,validation_split=0.5)

model.save('shape-pred.model')