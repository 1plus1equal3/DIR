import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MaxPooling, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.pool(x)
    
class AveragePooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(AveragePooling, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        return self.pool(x) 