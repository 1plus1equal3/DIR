import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from .blocks import BLOCKS
from .connector import CONNECTOR
from .blocks.convolution.conv import SimpleConvBlock 
from .config.mamba_unet import mamba_unet_encoder, mamba_unet_decoder
from torchsummary import summary
class MambaUNetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        self.encoder_connectors = nn.ModuleList()
        
        cfg_enc = mamba_unet_encoder
        block_type_enc = BLOCKS[cfg_enc['layer_block']['type']] 
        params_enc = cfg_enc['layer_block']['params']
        
        connector_type_enc = CONNECTOR[cfg_enc['layer_connector']['type']] 
        params_conn_enc = cfg_enc['layer_connector']['params']
        
        for i in range(cfg_enc['layer_num']):
            in_c = params_enc['in_channels'][i]
            out_c = params_enc['out_channels'][i]
            state_dim = params_enc['state_dim']
            layer = nn.Sequential(
                SimpleConvBlock(in_c, out_c, kernel_size=3, padding=1),
                block_type_enc(hidden_dim=out_c, state_dim=min(state_dim, out_c))
            )
            self.encoder_layers.append(layer)
            if i < cfg_enc['layer_num'] - 1:
                self.encoder_connectors.append(connector_type_enc(**params_conn_enc))

        self.decoder_layers = nn.ModuleList()
        self.decoder_connectors = nn.ModuleList()
        
        cfg_dec = mamba_unet_decoder
        block_type_dec = BLOCKS[cfg_dec['layer_block']['type']] 
        params_dec = cfg_dec['layer_block']['params']
        
        connector_type_dec = CONNECTOR[cfg_dec['layer_connector']['type']] 
        params_conn_dec = cfg_dec['layer_connector']['params']

        for i in range(cfg_dec['layer_num']):
            in_c = params_dec['in_channels'][i]
            out_c = params_dec['out_channels'][i]
            state_dim = params_dec['state_dim']

            self.decoder_connectors.append(connector_type_dec(**params_conn_dec))
            layer = nn.Sequential(
                SimpleConvBlock(in_c, out_c, kernel_size=3, padding=1),
                block_type_dec(hidden_dim=out_c, state_dim=min(state_dim, out_c))
            )
            self.decoder_layers.append(layer)
        final_in_c = cfg_dec['layer_block']['params']['out_channels'][-1]
        self.out_conv = nn.Conv2d(final_in_c, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x)
            
            if i < len(self.encoder_layers) - 1:
                skip_connections.append(x)
                x = self.encoder_connectors[i](x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.decoder_layers)):
            x = self.decoder_connectors[i](x) 
            skip = skip_connections[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_layers[i](x)
        return self.out_conv(x)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"cuda: {device}")
    model = MambaUNetModel(in_channels=3, out_channels=1).to(device)
    
    print(f"Ok")
    print("\n" + "=" * 70)
    print("           MODEL SUMMARY (Input: 3 x 256 x 256)  ")
    print("=" * 70)
    summary(model, (3, 128, 128))
    print("=" * 70 + "\n")


    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    print(f"Input shape:  {dummy_input.shape}")
    
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 1, 128, 128)
    print("Done")