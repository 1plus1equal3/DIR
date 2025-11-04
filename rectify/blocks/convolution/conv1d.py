import torch.nn as nn

class ConvLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, norm='bn', act_layer='relu'):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")
        
        if act_layer == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_layer is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation type: {act_layer}")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x