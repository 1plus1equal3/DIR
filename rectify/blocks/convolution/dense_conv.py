import torch 
import torch.nn as nn
import torch.nn.functional as F
from .dense_layer import DenseLayer 
class DenseBlock(nn.Module): 
    def __init__(self, nb_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()

        for i in range(nb_layers):
            layer_in_channels = in_channels + i * growth_rate
            layer = DenseLayer(
                layer_in_channels,
                growth_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            concatenated_features = torch.cat(features, 1)
            new_features = layer(concatenated_features)
            features.append(new_features)
            
        return torch.cat(features, 1)