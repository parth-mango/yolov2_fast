import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvblock(nn.Module):
  def __init__(self, input_channels, output_channels, size):
    super(DWConvblock, self).__init__()
    self.size= size
    self.input_channels= input_channels
    self.output_channels= output_channels

    