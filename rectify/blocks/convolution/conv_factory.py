import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFactory(nn.Module):
    def __init__(self, in_channels, nb_filter, dropout_rate=0.0):
        super(ConvFactory, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels, eps=1.1e-5)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels, nb_filter,
            kernel_size=3,
            stride=1,
            padding=1,  
            bias=False
        )
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    def forward(self, x):
        out = self.conv(self.act(self.norm(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        return out
