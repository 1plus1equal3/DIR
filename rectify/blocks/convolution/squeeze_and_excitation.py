import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        self.num_channels = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, num_channels, H, W = x.size()
        squeeze_tensor = x.view(batch_size, num_channels, -1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = fc_out_2.size()
        scale = fc_out_2.view(a, b, 1, 1)
        return x * scale
    
class SpatialSELayer(nn.Module):
    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, weights=None):
        
        batch_size, channel, a, b = x.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)
        squeeze_tensor = self.sigmoid(out).view(batch_size, 1, a, b)
        output = x * squeeze_tensor
        return output

class ChannelSpatialSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
    
    def forward(self, x):
        cse_out = self.cSE(x)
        sse_out = self.sSE(x)
        return torch.max(cse_out, sse_out)