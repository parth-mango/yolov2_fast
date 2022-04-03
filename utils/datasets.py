import os
import cv2
import random
import numpy as names

import torch
from torch.utils import data
from torch.utils.data import Dataset

# path= os.path.exists('data/small_coco/smallcoco.txt')

class TensorDataset():
#   print(os.path.exists('data/small_coco/smallcoco.txt'))
  def __init__(self, path, img_size_width= 352, img_size_height= 352,imgaug= False):
    
    assert os.path.exists(path)," %s The file path is wrong or does not exist" % path

    self.path= path
    self.data_list= []
    self.img_size_width= img_size_width
    self.img_size_height= img_size_height
    self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']
    self.imgaug= imgaug
    self.get_item()

    #Data check
  def data_check(self):
    with open(self.path, 'r') as f:
      for line in f.readlines():
        data_path= line.strip()
        # print(data_path, "datasets30")
        if os.path.exists(data_path):
          img_type= data_path.split(".")[-1]
          # print(img_type, "line 33 datasets.py")
          if img_type not in self.img_formats:
            raise Exception("img type error:%s" % img_type)
          else:
            self.data_list.append(data_path)
        else: 
          raise Exception("%s does not exist" % data_path)
    return self.data_list    
 
  
  def get_item(self):
    self.data_list= self.data_check()
    img_path= self.data_list[0]
    print(img_path)
  #   label_path= img_path.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
  #   print(label_path)    
    # print(self.data_list[0], "datasets")
       