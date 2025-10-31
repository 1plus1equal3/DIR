import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConvBlock 
class ResidualConvBlock(BaseConvBlock):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualConvBlock, self).__init__()
        NormLayer = nn.BatchNorm2d
        ActLayer = nn.ReLU 
        bias = False
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)
        self.norm1 = NormLayer(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm2 = NormLayer(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm3 = NormLayer(out_channels)

        self.activation = ActLayer(inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
                NormLayer(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out += shortcut
        out = self.activation(out)
        
        return out