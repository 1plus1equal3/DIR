from .base import BaseConvBlock
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(BaseConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, normalization='bn', norm_params={}, activation='relu', act_params={}):
        super(ConvBlock, self).__init__()
        bias = (normalization != 'bn')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.normalization = normalization
        
        if normalization not in self.norm_dict.keys():
            raise ValueError(f"Unsupported normalization type: {normalization}")
        else:
            self.norm = self.norm_dict[normalization](**norm_params)
            
        if activation not in self.act_dict():
            raise ValueError(f"Unsupported activation type: {activation}")
        else:
            self.activation = self.act_dict[activation](**act_params)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x