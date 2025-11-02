import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, bias=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=bias)
        )
    
    def forward(self, x):
        max_pool = self.max_pool(x)
        avg_pool = self.avg_pool(x)
        B, C, _, _ = x.size()
        linear_max = self.fc(max_pool.view(B, C)).view(B, C, 1, 1)
        linear_avg = self.fc(avg_pool.view(B, C)).view(B, C, 1, 1)
        attention = torch.sigmoid(linear_max + linear_avg)
        output = x * attention
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3, dilation=1, bias=False):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, dim=1).unsqueeze(1)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        output = x * attention
        return output

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7, padding=3, dilation=1, bias=True):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size, padding, dilation, bias)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out

# Example usage
if __name__ == "__main__":
    layer = CBAM(channel=64, reduction=16, kernel_size=7, padding=3)
    sample = torch.randn(2, 64, 32, 32)
    output = layer(sample)
    print(output.shape)