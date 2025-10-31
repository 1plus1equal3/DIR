#################################### Model configurations ####################################
## Simple Convolution UNet Encoder
unet_encoder = {
    "layer_num": 4,
    "layer_block": {
        "type": "conv",  # Options:
        "params": {
            "in_channels": [3, 64, 128, 256],
            "out_channels": [64, 128, 256, 512],
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "use_norm": True,
            "use_act": True
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
}

## Simple Convolution UNet Decoder
unet_decoder = {
    "layer_num": 4,
    "layer_block": {    
        "type": "conv",  # Options:
        "params": {
            "in_channels": [512, 256, 128, 64],
            "out_channels": [256, 128, 64, 3],
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "use_norm": True,
            "use_act": True
        }
    },
    "block_num": [4, 3, 3, 2],  # Number of blocks in each layer
    "layer_connector": {
        "type": "upsampling",
        "params": {
            "scale_factor": 2,
            "mode": "nearest",
        }
    },
}

unet_training = {
    "loss_function": [
        {
            "type": "l1",  # Options: 'l1', 'l2'
            "params": {}
        }
    ],
    "optimizer": "adam"
}