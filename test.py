import cv2
import tensorflow as tf
import sys
import numpy as np

categories = ["triangle","circle","star"]

def prep(path):
  SIZE = 75
  img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  new_arr = cv2.resize(img_arr,(SIZE,SIZE))
  new_arr = np.array(new_arr)/255.0
  new_arr = new_arr.reshape(1,SIZE,SIZE)
  return new_arr

model = tf.keras.models.load_model("shape-pred.model")

prediction = model.predict(prep(sys.argv[1]))

prediction = list(prediction)

print("predictions:")

print(f"triangle: {prediction[0][0]:.20f}")
print(f"circle: {prediction[0][1]:.20f}")
print(f"star: {prediction[0][2]:.20f}")