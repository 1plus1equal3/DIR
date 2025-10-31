import torch
import torch.nn as nn

class DConvBlock(nn.Module):
    def __init__(self, inshape, dim=64, expansion_factor=1.0, bias=False):
        super(DConvBlock, self).__init__()
        hidden_features = int(dim*expansion_factor)
        self.conv = nn.Conv2d(inshape, hidden_features, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.depthwise(x)
        return x
    
# Example usage
if __name__ == "__main__":
    layer = DConvBlock(inshape=32, dim=64, expansion_factor=2.0)
    input_tensor = torch.randn(1, 32, 128, 128)  # Example input
    output_tensor = layer(input_tensor)
    print(output_tensor.shape)  # Should print torch.Size([1, 128, 128, 128])