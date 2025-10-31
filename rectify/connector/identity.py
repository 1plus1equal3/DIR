import torch.nn as nn

class Indentity(nn.Module):
    def __init__(self):
        super(Indentity, self).__init__()

    def forward(self, x):
        return x