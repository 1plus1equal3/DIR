import torch.nn as nn

class Upsampling(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.Upsample(scale_factor=self.scale_factor, mode=self.mode)(x)