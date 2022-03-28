import os
import cv2
import random
import numpy as names

import torch
from torch.utils import data
from torch.utils.data import Dataset

class TensorDataset():
  def __init__(self, path, img_size_width= 352, img_size_height= 352,imgaug= False):
    assert os.path.exists(path)," %s The file path is wrong or does not exist" % path

    self.path= path
    self.data_list= []
    self.img_size_width= img_size_width
    self.img_size_height= img_size_height
    self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']
    self.imgaug= imgaug

    #Data check
    with open(self.path, 'r') as f:
      for line in f.readlines():
        data_path= line.strip()
        if os.path.exists(data_path):
          img_type= data_path.split(".")[-1]