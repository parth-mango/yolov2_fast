import math
import torch
import torch.nn as nn
import numpy as np

def build_target(preds, targets, cfg, device):
  tcls, tbox, indices, anch= [], [], [], []
  
  #anchor box Quantity, the number of labels in the current batch
  anchor_num, label_num= cfg["anchor_num"], targets.shape[0]
  
  #Load anchor configuration

  anchors= np.array(cfg["anchors"])
  anchors= torch.from_numpy(anchors.reshape(len(preds)// 3, anchor_num, 2)).to(device)
  
  gain = torch.ones(7, device= device)

  at= torch.arange(anchor_num, device= device).float().view(anchor_num, 1).repeat(1, label_num)
  

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







