import torch 
import torch.nn as nn
import torch.nn.functional as F

# Kế thừa thẳng từ nn.Module, vì không dùng BaseConvBlock nữa
class DenseLayer(nn.Module): 
    def __init__(self, in_channels, growth_rate, bias=False):
        super(DenseLayer, self).__init__()
        self.bias = bias
        inter_channels = 4 * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        return out