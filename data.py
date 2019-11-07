import numpy as np
import cv2
import os
from pathlib import Path

class AryllaDataHandler(object):
  
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.data = self._get_data()
  
  def _get_data(self):

    if self.is_train:
      Image_path = 'C:/Users/xwen2/Desktop/DIRNet-Deformable image registration/Data2/Training/'
      Label_path = 'C:/Users/xwen2/Desktop/DIRNet-Deformable image registration/Label/training_label.txt'
    else :
      Image_path = 'C:/Users/xwen2/Desktop/DIRNet-Deformable image registration/Data2/Testing/'
      Label_path = 'C:/Users/xwen2/Desktop/DIRNet-Deformable image registration/Label/testing_label.txt'
    
    images = []  # ndarray
    labels = []

    for filename in os.listdir(Image_path):
      img = cv2.imread(Image_path + filename)
      #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      img = cv2.resize(img, (40, 40), interpolation = cv2.INTER_AREA)
      images.append(img)
    images = np.asarray(images)
    images = np.expand_dims(images, axis=1)

    f = open(Label_path, 'r')
    lines = f.readlines()
    for line in lines:
      if line is not "":
        line = line.split(',')
        labels.append(line[1])
    f.close()
    labels = np.asarray(labels, dtype=np.uint8)

    values, counts = np.unique(labels, return_counts=True)

    data = []
    
    for i in range(2):
      label = values[i]
      count = counts[i]
      arr = np.empty([count, 40, 40, 3], dtype=np.float32)
      data.append(arr)

    l_iter = [0] * 2

    for i in range(labels.shape[0]):
      label = labels[i]
      
      data[label][l_iter[label]] = images[i] / 255.
      l_iter[label] += 1

    return data

  def sample_pair(self, batch_size, label=None):
    
    label = np.random.randint(2) if label is None else label
    images = self.data[label]
    
    choice1 = np.random.choice(images.shape[0], batch_size)
    choice2 = np.random.choice(images.shape[0], batch_size)
    x = images[choice1]
    y = images[choice2]

    return x, y

  def get_pair_by_idx(self, idx, batch_size=1):
    x = self.s_data[np.expand_dims(idx, 0)]
    y = self.d_data[np.expand_dims(idx, 0)]
    labels = self.labels[np.expand_dims(idx, 0)]
        
    return x, y, labels

  def get_eval_pair_by_idx(self, idx, batch_size=1):
        
    x = self.s_data_eval[np.expand_dims(idx, 0)]
    y = self.d_data_eval[np.expand_dims(idx, 0)]
    labels = self.labels_eval[np.expand_dims(idx, 0)]
    return x, y, labels

