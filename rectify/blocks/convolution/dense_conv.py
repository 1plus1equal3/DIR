import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConvBlock 


class _DenseLayer(BaseConvBlock):
    def __init__(self, in_channels, growth_rate, normalization='bn', norm_params={}, activation='relu', act_params={}):
        super(_DenseLayer, self).__init__()
        if normalization not in self.norm_dict.keys():
            raise ValueError(f"Unsupported normalization type: {normalization}")
        if activation not in self.act_dict.keys():
            raise ValueError(f"Unsupported activation type: {activation}")
            
        NormLayer = self.norm_dict[normalization]
        ActLayer = self.act_dict[activation]
        bias = False 

        inter_channels = 4 * growth_rate

        self.norm1 = NormLayer(in_channels, **norm_params)
        self.act1 = ActLayer(**act_params)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
        self.norm2 = NormLayer(inter_channels, **norm_params)
        self.act2 = ActLayer(**act_params)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        
        return out

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_channels, growth_rate, 
                 normalization='bn', norm_params={}, activation='relu', act_params={}):

        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(nb_layers):
            layer_in_channels = in_channels + i * growth_rate
            layer = _DenseLayer(
                layer_in_channels, 
                growth_rate, 
                normalization, 
                norm_params, 
                activation, 
                act_params
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            concatenated_features = torch.cat(features, 1)
            new_features = layer(concatenated_features)
            features.append(new_features)
        return torch.cat(features, 1)