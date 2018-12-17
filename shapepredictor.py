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

CATEGORIES = ["triangle","circle","star"]


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

x = np.array(x)
x = x/255.0

model = Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(75,75)))

model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(len(CATEGORIES), activation=tf.nn.softmax))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=10)

model.save('shape-pred.model')