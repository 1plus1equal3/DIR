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
LOSS_FUNCTION = [
    {
        "type": "l1",  # Options: 'l1', 'l2'
        "params": {},
        "weight": 1,
    }
]

OPTIMIZER = 'adam'

## Training configurations
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

## Checkpoint configurations
CHECKPOINT_DIR = "./checkpoints"
MAX_CHECKPOINTS = 10
SAVE_EVERY_N_EPOCHS = 5
MODEL_NAME = "rectify_unet"

## Logging configurations
WANDB_PROJECT = "document_rectification"
WANDB_API_KEY_PATH = "./wandb_key.txt"
LOG_EVERY_N_STEPS = 10
VISUALIZE_EVERY_N_EPOCHS = 1
NUM_VIS_SAMPLES = 4

## Validation configurations
VAL_EVERY_N_EPOCHS = 1