import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_factory import ConvFactory


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, layers_in_block, drop_rate=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        num_channels = in_channels
        for _ in range(layers_in_block):
            self.layers.append(ConvFactory(num_channels, growth_rate, drop_rate))
            num_channels += growth_rate

        self.out_channels = num_channels  
    def forward(self, x):
        features = [x] 
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        out = torch.cat(features, dim=1)
        return out

