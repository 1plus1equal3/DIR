import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *
from .blocks import BLOCKS
from .connector import CONNECTOR
from .losses import LOSS_FUNCTIONS

class CustomUNet(nn.Module):
    """ 
    Customizable UNet model based on configuration parameters.
    """
    def __init__(self):
        super(CustomUNet, self).__init__()
        self.layer_num = LAYER_NUM
        self.layer_block = LAYER_BLOCK
        self.block_num = BLOCK_NUM
        self.layer_connector = LAYER_CONNECTOR
        
    def build_block(self, params):
        block = BLOCKS[self.layer_block['type']]
        return block(**params)
    
    def build_layer(self, in_channels, out_channels, block_params, block_num):
        layer = []
        for _ in range(block_num):
            layer.append(self.build_block(block_params))
    
    def build_connector(self, params):
        connector = CONNECTOR[self.layer_connector['type']]
        return connector(**params)
    


