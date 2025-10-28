import torch
import torch.nn as nn
#################################### Model configurations ####################################
## Model architecture
LAYER_NUM = 4
LAYER_BLOCK = {
    "type": "convolution",  # Options: 'convolution', 'transformer'
    "params": {
        "in_channels": [3, 64, 128, 256],
        "out_channels": [64, 128, 256, 512],
        "kernel_size": 3,
    }
}
BLOCK_NUM = [2, 3, 3, 4] # Number of blocks in each layer
LAYER_CONNECTOR = {
    "type": "max_pooling",
    "params": {
        "kernel_size": 2,
        "stride": 2,
    }
}

## Model training
LOSS_FUNCTION = {
    {
        "type": "l1",  # Options: 'l1', 'l2'
        "params": {}
    }
}

OPTIMIZER = 'adam'