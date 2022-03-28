import torch
import torchvision
import torch.nn.functional as F

import os, time
import numpy as np
from tqdm import tqdm


def load_datafile(data_path):
  cfg= {"model_name":None,
        "epochs": None,
        "steps": None,
        "batch_size":None,
        "subdivisions": None,
        "learning_rate":None,

        "pre_weights": None,
        "classes": None, 
        "width": None,
        "height": None,
        "anchor_num": None,
        "anchors": None,

        "val": None,
        "train": None,
        "names": None
        }
  assert os.path.exists(data_path), "Please specify the correct configuration .data file path"

  #Specify the type of configuration item
  list_type_key= ["anchors", "steps"]
  str_type_key= ["model_name", "val", "train", "names", "pre_weights"]
  int_type_key= ["epochs", "batch_size", "classes", "width", "height",
                 "anchor_num", "subdivisions"]
  float_type_key= ["learning_rate"]

  #load configuration file               
  with open(data_path, 'r') as f:
    for line in f.readlines():
      if line == '\n' or line[0] == "[" :
        continue
      else:
        data= line.strip().split("=")
        #Configuration item type conversion
        if data[0] in cfg:
          if data[0] in int_type_key:
            cfg[data[0]] = int(data[1])
          elif data[0] in str_type_key:
            cfg[data[0]]= data[1]
          elif data[0] in float_type_key:
            cfg[data[0]]= float(data[1])
          elif data[0] in list_type_key:
            cfg[data[0]] = [float(x) for x in data[1].split(",")]
          else:
            print("The configuration file has the wrong configuration item")
        
        else:
          print("%sThere is an invalid configuration item in the configuration file:%s"%(data_path, data)) 
                     
  return cfg