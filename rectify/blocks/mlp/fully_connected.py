import torch.nn as nn
from ..convolution import SimpleConvBlock

class FullyConnectedFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class FullyConnectedFFN_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedFFN_v2, self).__init__()
        self.fc1 = SimpleConvBlock(input_dim, hidden_dim, 1, stride=1, padding=0, bias=False, use_norm=True, use_act=True)
        self.fc2 = SimpleConvBlock(hidden_dim, output_dim, 1, stride=1, padding=0, bias=False, use_norm=False, use_act=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x