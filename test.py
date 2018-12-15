import cv2
import tensorflow as tf
import sys


categories = ["circle","triangle","star"]

def prep(path):
  size = 50
  img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  new_arr = cv2.resize(img_arr,(size,size))
  return new_arr.reshape(-1,size,size,1)

model = tf.keras.models.load_model("shape-pred.model")

prediction = model.predict(prep(sys.argv[1]))

print(prediction[0][0])

print(categories[int(prediction[0][0])])