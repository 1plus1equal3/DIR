import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention2D(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(Attention2D, self).__init__()
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.shape[0], -1)
        return F.softmax(x / self.temperature, dim=1)
    

class Dynamic_Conv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_Conv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.ratio = ratio
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = Attention2D(in_planes, ratios=ratio, K=K, temperature=temperature, init_weight=init_weight)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for i in range (self.K):
            nn.init.kaiming_uniform_(self.weight[i])
    
    def update_temperature(self):
        self.attention.update_temperature()
    
    def forward(self, x):
        softmax_attn = self.attention(x)
        B, C, H, W = x.shape
        x = x.view(1, -1, H, W)
        weight = self.weight.view(self.K, -1)
        aggregated_weight = torch.mm(softmax_attn, weight).view(B*self.out_planes, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregated_bias = torch.mm(softmax_attn, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregated_weight, bias=aggregated_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*B)
        else:
            output = F.conv2d(x, weight=aggregated_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*B)
        output = output.view(B, self.out_planes, output.shape[-2], output.shape[-1])
        return output

# Example usage
if __name__ == "__main__":
    layer = Dynamic_Conv2D(in_planes=64, out_planes=128, kernel_size=3)
    sample = torch.randn(2, 64, 32, 32)
    output = layer(sample)
    print(output.shape)