import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConvBlock 

class TransitionConvBlock(BaseConvBlock):
    def __init__(self, in_channels, out_channels, normalization='bn', norm_params={}, activation='relu', act_params={}):
        super(TransitionConvBlock, self).__init__()
        if normalization not in self.norm_dict.keys():
            raise ValueError(f"Unsupported normalization type: {normalization}")
        if activation not in self.act_dict.keys():
            raise ValueError(f"Unsupported activation type: {activation}")
        self.norm = self.norm_dict[normalization](in_channels, **norm_params)
        self.activation = self.act_dict[activation](**act_params)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = 0.0
    def forward(self, x):
        out = self.conv1(self.activation(self.norm(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return F.avg_pool2d(out, 2)