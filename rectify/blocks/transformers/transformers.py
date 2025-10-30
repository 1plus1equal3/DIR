import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward,
                                                            dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x.permute(1, 0, 2)  # Transformer expects input of shape (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_length, d_model)
        return x