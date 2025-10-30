from .base import BaseConvBlock
import torch.nn as nn

class ConvBlock(BaseConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, normalization='bn', norm_params={}, activation='relu', act_params={}):
        super(ConvBlock, self).__init__()
        bias = (normalization != 'bn')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
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
    
class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_norm=True, use_act=True):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if use_norm:
            self.use_norm = use_norm
            self.norm = nn.BatchNorm2d(out_channels)
        if use_act:
            self.use_act = use_act
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm is not None:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x