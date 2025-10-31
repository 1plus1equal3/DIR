import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConvBlock 
class TransitionConvBlock(BaseConvBlock):
    def __init__(self, in_channels, out_channels, droprate=0.0):
        super(TransitionConvBlock, self).__init__()
        NormLayer = nn.BatchNorm2d
        ActLayer = nn.ReLU
        self.norm = NormLayer(in_channels)
        self.activation = ActLayer(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.droprate = droprate
        
    def forward(self, x):
        out = self.conv1(self.activation(self.norm(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return F.avg_pool2d(out, 2)