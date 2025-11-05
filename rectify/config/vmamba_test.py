#################################### Model configurations ####################################
## VMamba Test Model - Simple architecture with VMamba blocks

model = {
    "layer_num": 3,
    "layer_block": {
        "type": "vmamba",
        "params": {
            "in_channels": [3, 64, 128],
            "out_channels": [64, 128, 256],
            "d_state": 16,              # SSM state dimension
            "d_inner_multiplier": 2.0,  # d_inner = in_channels * 2
            "dt_rank": "auto",          # auto = in_channels // 16
            "d_conv": 3,                # depthwise conv kernel size
            "scan_mode": 0,             # 0: cross2d, 1: unidi, 2: bidi, 3: rot90
        }
    },
    "block_num": [1, 1, 1],  # Number of blocks in each layer
    "layer_connector": {
        "type": "max_pooling",
        "params": {
            "kernel_size": 2,
            "stride": 2,
        }
    },
}

## Hybrid Model - Mix VMamba and Conv blocks
hybrid_model = {
    "layer_num": 4,
    "layer_block": {
        "type": ["conv", "vmamba", "vmamba", "conv"],  # Mix of block types
        "params": {
            "in_channels": [3, 64, 128, 256],
            "out_channels": [64, 128, 256, 512],
            "kernel_size": [3, None, None, 3],  # Only for conv blocks
            "stride": [1, None, None, 1],
            "padding": [1, None, None, 1],
            "use_norm": [True, None, None, True],
            "use_act": [True, None, None, True],
            "d_state": [None, 16, 16, None],  # Only for vmamba blocks
            "d_inner_multiplier": [None, 2.0, 2.0, None],
            "scan_mode": [None, 0, 0, None],
        }
    },
    "block_num": [2, 1, 1, 2],
    "layer_connector": {
        "type": "max_pooling",
        "params": {
            "kernel_size": 2,
            "stride": 2,
        }
    },
}

training = {
    "loss_function": [
        {
            "type": "l1",
            "params": {}
        }
    ],
    "optimizer": "adam"
}
