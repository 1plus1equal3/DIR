import torch.nn as nn

class BaseConvBlock(nn.Module):
    """Base class for convolutional blocks."""
    def __init__(self):
        super(BaseConvBlock, self).__init__()
        self.norm_dict = {
            'bn': nn.BatchNorm2d,
            'in': nn.InstanceNorm2d,
            'gn': nn.GroupNorm,
            'none': None
        }
        self.act_dict = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'tanh': nn.Tanh,
            'none': None
        }
    
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")