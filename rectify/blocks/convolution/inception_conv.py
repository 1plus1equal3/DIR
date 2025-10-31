import torch
import torch.nn as nn
from .base import BaseConvBlock
from .conv import SimpleConvBlock

class InceptionBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels
    ):
        super(InceptionBlock, self).__init__()
        self.branch_1 = SimpleConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self.branch_2 = SimpleConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.branch_3 = SimpleConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        self.project = SimpleConvBlock(
            out_channels * 3, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(x)
        branch3 = self.branch_3(x)
        outputs = torch.cat([branch1, branch2, branch3], dim=1)
        outputs = self.project(outputs)
        return outputs
    
class InceptionBlock_v2(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
    ):
        super(InceptionBlock_v2, self).__init__()
        self.branch_1 = SimpleConvBlock(
            in_channels,
            out_channels // 4,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.branch_2 = nn.Sequential(
            SimpleConvBlock(
                in_channels,
                out_channels  // 4,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            SimpleConvBlock(
                out_channels  // 4,
                out_channels  // 4,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self.branch_3 = nn.Sequential(
            SimpleConvBlock(
                in_channels,
                out_channels  // 4,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            SimpleConvBlock(
                out_channels,
                out_channels  // 4,
                kernel_size=5,
                stride=1,
                padding=2
            )
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            SimpleConvBlock(
                in_channels,
                out_channels  // 4,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(x)
        branch3 = self.branch_3(x)
        branch4 = self.branch_4(x)
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs
       
    
# Example usage
if __name__ == "__main__":
    layer = InceptionBlock(in_channels=64, out_channels=64)
    sample = torch.randn(1, 64, 28, 28)
    output = layer(sample)
    print(output.shape)