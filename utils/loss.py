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

  # We are concatenating anchor number as another dimension to the three repeats of the target 


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

        # anchor iou match
        r=  gt[:, : , 4: 6 ] / anchors_cfg[:, None]  # we are comparing the target(image) w&h with anchor(bounding box) w&h -- normalisation ??

        j= torch.max(r, 1.0/r).max(2)[0] < 2 # here first we find the max between r and 1/r - element wise comparision of tensor, then we find max of 
                                             # the tensor wrt 3rd dimension(denoted by .max(2)) and this return a tuple with first element as value and                                         
                                             # second element as index. Finally we compare the value if it is less than 2.
        t= gt[j] # Only those targets with wh ratio less than 2 are retained 
        
        #Expand dimension and copy data                                 
        #offsets
        gxy= t[:, 2:4] # grid xy
        gxi = gain[[2, 3]] - gxy  # inverse - from the other side of origin

        j,k = ((gxy % 1. < g ) & (gxy > 1.)).T # Rounding off?
        l, m= ((gxi % 1. < g) & (gxi >1.)).T
        j= torch.stack((torch.ones_like(j), j, k, l, m)) # stacking "All true value with dim = dim(j)" along with j, k, l,m"
        t= t.repeat((5, 1, 1 ))[j] # We keep only those targets which have decimal part less than 0.5 either for w or h
        offsets= (torch.zeros_like(gxy)[None] + off[:, None])[j]
        
      else:
        t= targets[0]
        offsets= 0
      
      #define 
      b, c= t[: , :2].long().T#image and class
      gxy= t[:, 2:4] # grid xy
      gwh= t[:, 4:6] #grid wh
      gij= (gxy- offsets).long() # Here we are rounding the gxy on the basis of offsets
      
      gi, gj = gij.T # grid xy indices
      
      # Append
      a= t[:, 6].long()
      
      indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))) # clamping the coordinates within the final prediction sizes
      
      tbox.append(torch.cat((gxy-gij, gwh), 1)) # Here we are finding the x and y coordinates wrt 1x1 square among the 22x22 of the entire output size
      
      anch.append(anchors_cfg[a]) # anchors
      tcls.append(c) #class
  return tcls, tbox, indices, anch


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
  print(tcls)






