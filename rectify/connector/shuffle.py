import torch.nn as nn

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor=2):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(x)
    
class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor=2):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)

    def forward(self, x):
        return self.pixel_unshuffle(x)