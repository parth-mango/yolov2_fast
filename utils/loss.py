import math
import torch
import torch.nn as nn
import numpy as np

layer_index = [0, 0, 0, 1, 1, 1]

def build_target(preds, targets, cfg, device):
  tcls, tbox, indices, anch= [], [], [], []
  
  #anchor box Quantity, the number of labels in the current batch
  anchor_num, label_num= cfg["anchor_num"], targets.shape[0]    # 3, variable shape of targets(x)
  
  #Load anchor configuration

  anchors= np.array(cfg["anchors"])
  anchors= torch.from_numpy(anchors.reshape(len(preds)// 3, anchor_num, 2)).to(device)
  
  gain = torch.ones(7, device= device)

  at= torch.arange(anchor_num, device= device).float().view(anchor_num, 1).repeat(1, label_num) # shape =  3, x, 6
  
  targets= torch.cat((targets.repeat(anchor_num, 1, 1), at[:, :, None]), 2)  # 3, x , 7 --> 0, 1, 2 added to each array as last element
  
  g= 0.5 # bias

  off= torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], ], device= device).float() * g #offsets
  

  for i, pred in enumerate(preds):

    if i % 3 == 0:
      _, _, h, w = pred.shape # 22, 22 and 11, 11

      assert cfg["width"]/w == cfg["height"]/h, "Inconsistent downsampling of feature map width and height"
      
      # Calculate the downsampling multiple
      
      stride = cfg["width"]/w
      
      #The anchor configuration corresponding to the feature map of this scale      
      anchors_cfg = anchors[layer_index[i]]/stride # scaling the anchor values down

      #Map the label coordinates to the feature map
      gain[2:6] = torch.tensor(pred.shape)[[3, 2, 3, 2]]

      gt = targets * gain # scaling 

      if label_num :
        

def smooth_BCE(eps= 0.1):
  #return positive, negative label smoothing BCE targets
  return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(preds, targets, cfg, device):
  balance= [1.0, 0.4]

  ft= torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
  lcls, lbox, lobj= ft([0]), ft([0]), ft([0])  #currently tensor with value as Zero

  #Define the loss function of obj and cls
  BCEcls= nn.CrossEntropyLoss()
  BCEobj= nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device= device))

  cp,cn= smooth_BCE(eps= 0.0)

  #build gt
  tcls, tbox, indices, anchors= build_target(preds, targets, cfg, device)







