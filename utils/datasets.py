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

    #Data check
    with open(self.path, 'r') as f:
      for line in f.readlines():
        data_path= line.strip()
        
        if os.path.exists(data_path):
          img_type= data_path.split(".")[-1]
          if img_type not in self.img_formats:
            raise Exception("img type error:%s" % img_type)
          else:
            self.data_list.append(data_path)
        else: 
          raise Exception("%s does not exist" % data_path)

  def __getitem__(self, index):
    img_path= self.data_list[index]
    print(img_path)
    # label_path= img_path.split(".")[0] + ".txt"          

          