#################################### Model configurations ####################################
## Simple Convolution Model
sample_config = {
    "model": {
        "layer_num": 4,
        "layer_block": {
            "type": "conv",  # Options:
            "params": {
                "in_channels": [3, 64, 128, 256],
                "out_channels": [64, 128, 256, 512],
                "kernel_size": 3,
            }
        },
        "block_num": [2, 3, 3, 4],  # Number of blocks in each layer
        "layer_connector": {
            "type": "max_pooling",
            "params": {
                "kernel_size": 2,
                "stride": 2,
            }
        },
    },
    "training": {
        "loss_function": [
            {
                "type": "l1",  # Options: 'l1', 'l2'
                "params": {}
            }
        ],
        "optimizer": "adam"
    }
}