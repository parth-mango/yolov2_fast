import os
import cv2
import random
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import Dataset


def contrast_and_brightness(img):
  alpha= random.uniform(0.25, 1.75)
  beta= random.uniform(0.25, 1.75)
  blank= np.zeros(img.shape, img.dtype)
  #dst= alpha * img + beta * blank
  dst= cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
  return dst

def motion_blur(image):
  if random.randint(1, 2) == 1:
    degree= random.randint(2, 3)
    angle= random.uniform(-360, 360)
    image= np.array(image)
  
    # Here, a matrix of motion blur kernels at any angle is generated. The larger the degree, the higher the degree of blurring.
    M= cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel= np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel= motion_blur_kernel / degree
    blurred= cv2.filter2D(image, -1, motion_blur_kernel)
    
    #convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINIMAX)
    blurred= np.array(blurred, dtype= np.uint8)
    return blurred

  else:
    return image

def augment_hsv(img, hgain = 0.0138, sgain= 0.678, vgain= 0.36):
  r= np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1 #random gain
  hue, sat, val= cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
  dtype= img.dtype #uint8
  

  x= np.arange(0, 256, dtype=np.int16)
  lut_hue= ((x * r[0]) % 180).astype(dtype)
  lut_sat= np.clip(x * r[1], 0, 255).astype(dtype)
  lut_val= np.clip(x * r[2], 0, 255).astype(dtype)

  img_hsv= cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
  img= cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) 
  return img


def img_aug(img):
  img= contrast_and_brightness(img)
  # img= motion_blur(img)
  # img= random_resize(img)
  # img= augment_hsv(img)
  return img


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
    # print(img_path)
    label_path= img_path.replace('images', 'labels').replace(os.path.splitext(img_path)[-1], '.txt')
    # print(label_path)    

    # normalization operation
    img= cv2.imread(img_path)
    img= cv2.resize(img, (self.img_size_width, self.img_size_height), interpolation= cv2.INTER_LINEAR)
    
    # data augmentation
    if self.imgaug == True:
      img= img_aug(img)
       