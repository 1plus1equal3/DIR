import torch.nn as nn
from copy import deepcopy
from blocks import BLOCKS
from connector import CONNECTOR

class CustomModel(nn.Module):
    """ 
    Customizable CustomModel model based on configuration parameters.
    """
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.config = config
        self.layer_num = config['layer_num'] # Number of layers in the model
        self.layer_block = deepcopy(config['layer_block']) # Block configuration for each layer
        self.block_num = deepcopy(config['block_num']) # Number of blocks in each layer
        self.layer_connector = deepcopy(config['layer_connector']) # Connector configuration between layers
        # Validate configuration parameters
        self.ultimate_validation()
        # Build the model
        self.model_layers = self.build_model()

    def ultimate_validation(self):
        """ 
        Validate configuration parameters for consistency.
        Number of blocks in each layer must match the number of layers.
        Block Validation:
            - Validate block type exists.
            - Block params must match the number of layers or be a single unique value.
        Connector Validation:
            - Validate connector type exists.
            - Connector params must match the number of layers - 1 or be a single unique value.
        """
        if len(self.block_num) != self.layer_num:
            print(f"block_num length {len(self.block_num)} does not match layer_num {self.layer_num}.")
            raise ValueError("Configuration block_num length mismatch.")
        # Validate block type
        if self.layer_block['type'] not in BLOCKS:
            print(f"Block type '{self.layer_block['type']}' is not recognized.")
            raise ValueError("Invalid block type in configuration.")
        # Validate connector type
        if self.layer_connector['type'] not in CONNECTOR:
            print(f"Connector type '{self.layer_connector['type']}' is not recognized.")
            raise ValueError("Invalid connector type in configuration.")
        # Validate block params
        block_params = self.layer_block['params']
        for k, v in block_params.items():
            if isinstance(v, list):
                if len(v) != self.layer_num:
                    print(f"Parameter '{k}' length {len(v)} does not match layer_num {self.layer_num}.")
                    raise ValueError("Configuration parameter length mismatch.")
            if not isinstance(v, list):
                block_params[k] = [v] * self.layer_num
        # Validate connector params
        connector_params = self.layer_connector['params']
        for k, v in connector_params.items():
            if isinstance(v, list):
                if len(v) != self.layer_num - 1:
                    print(f"Connector parameter '{k}' length {len(v)} does not match layer_num - 1 {self.layer_num - 1}.")
                    raise ValueError("Configuration connector parameter length mismatch.")
            if not isinstance(v, list):
                connector_params[k] = [v] * (self.layer_num - 1)
        
    def build_block(self, params):
        """
        Build a simple block based on the specified type and parameters.
        """
        block = BLOCKS[self.layer_block['type']]
        return block(**params)
    
    def build_layer(self, block_num, _params):
        """ 
        Build a layer composed of multiple blocks.
        NOTE: The first block uses the provided 'in_channels', subsequent blocks use the 'out_channels' of the first block as 'in_channels'.
        """
        layer = []
        params = _params.copy()
        out_channels = params['out_channels']
        layer.append(self.build_block(params))
        params['in_channels'] = out_channels
        for _ in range(1, block_num):
            layer.append(self.build_block(params))
        return nn.Sequential(*layer)
    
    def build_connector(self, params):
        """ 
        Build a connector between layers based on the specified type and parameters.
        """
        connector = CONNECTOR[self.layer_connector['type']]
        return connector(**params)
    
    def build_model(self):
        """
        Build the complete model based on configuration parameters.
        """
        model_layers = []
        block_params = self.layer_block['params']
        connector_params = self.layer_connector['params']
        for layer in range(self.layer_num):
            params = {k: v[layer] for k, v in block_params.items()}
            layer_module = self.build_layer(self.block_num[layer], params)
            model_layers.append(layer_module)
            if layer < self.layer_num - 1:
                params = {k: v[layer] for k, v in connector_params.items()}
                connector_module = self.build_connector(params)
                model_layers.append(connector_module)
        return nn.ModuleList(model_layers)
    
    def forward(self, x):
        """
        Forward pass through the model.
        """
        for layer in self.model_layers:
            x = layer(x)
        return x

# Test build model
if __name__ == "__main__":
    from config import sample_config
    model_config = sample_config['model']
    model = CustomModel(model_config)
    print(model)