import torch
import torch.nn as nn
from base import BaseConvBlock
from conv import ConvBlock

class InceptionBlock(BaseConvBlock):
    def __init__(
            self, 
            in_channels, 
            out_1x1, 
            outinception_3x3_reduced, 
            outinception_3x3, 
            outinception_5x5_reduced,
            outinception_5x5,
            out_pool,
            normalization='bn',
            norm_params={},
            activation='relu',
            act_params={}
    ):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1, normalization=normalization, norm_params=norm_params, activation=activation, act_params=act_params)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, outinception_3x3_reduced, kernel_size=1, normalization=normalization, norm_params=norm_params, activation=activation, act_params=act_params),
            ConvBlock(outinception_3x3_reduced, outinception_3x3, kernel_size=3, padding=1, normalization=normalization, norm_params=norm_params, activation=activation, act_params=act_params)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, outinception_5x5_reduced, kernel_size=1, normalization=normalization, norm_params=norm_params, activation=activation, act_params=act_params),
            ConvBlock(outinception_5x5_reduced, outinception_5x5, kernel_size=5, padding=2, normalization=normalization, norm_params=norm_params, activation=activation, act_params=act_params)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1, normalization=normalization, norm_params=norm_params, activation=activation, act_params=act_params)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        
        outputs = [branch1_out, branch2_out, branch3_out, branch4_out]
        return torch.cat(outputs, 1)
    
# Example usage
if __name__ == "__main__":
    layer = InceptionBlock(
        in_channels=192,
        out_1x1=64,
        outinception_3x3_reduced=96,
        outinception_3x3=128,
        outinception_5x5_reduced=16,
        outinception_5x5=32,
        out_pool=32,
        norm_params={'num_features': 192}
    )
    sample = torch.randn(1, 192, 28, 28)
    output = layer(sample)
    print(output.shape)