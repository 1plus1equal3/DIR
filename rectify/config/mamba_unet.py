mamba_unet_encoder = {
    "layer_num": 4,
    "layer_block": {
        "type": "mamba_ir",
        "params": {
            "in_channels": [3, 64, 128, 256],
            "out_channels": [64, 128, 256, 512],
            "state_dim": 64,
        }
    },
    "block_num": [1, 1, 1, 1], 
    "layer_connector": {
        "type": "max_pooling",
        "params": {
            "kernel_size": 2,
            "stride": 2,
        }
    },
}

mamba_unet_decoder = {
    "layer_num": 3,
    "layer_block": {    
        "type": "mamba_ir",
        "params": {
            # in_channels: (kênh từ skip) + (kênh từ upsample)
            # Tầng 1: (Skip 256) + (Up 512) = 768 -> Out 256
            # Tầng 2: (Skip 128) + (Up 256) = 384 -> Out 128
            # Tầng 3: (Skip 64)  + (Up 128) = 192 -> Out 64
            # Tầng 4: (Skip 3)   + (Up 64)  = 67  -> Out 32
            "in_channels": [512 + 256, 256 + 128, 128 + 64],
            "out_channels": [256, 128, 64],
            "state_dim": 64,
        }
    },
    "block_num": [1, 1, 1], 
    "layer_connector": {
        "type": "upsampling",
        "params": {
            "scale_factor": 2,
            "mode": "bilinear", 
        }
    },
}