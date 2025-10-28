import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConvBlock
class ResidualConvBlock(BaseConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, normalization='bn', norm_params={}, activation='relu', act_params={}):
        pass

    def forward(self, x):
        pass