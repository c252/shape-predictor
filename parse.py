import os
import cv2

data = "/home/cb/ml-projects/shape-predict"

SIZE = 50

def get_traindata(training_in,class_names):
  for i in class_names:
    path = os.path.join(data,i)
    class_nm = class_names.index(i)
    for img in os.listdir(path):
      img_arr = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
      train_arr = cv2.resize(img_arr, (SIZE,SIZE))

      training_in.append([train_arr,class_nm])