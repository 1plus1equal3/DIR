import torch
import torch.nn as nn
from .base import BaseLossFunction

class L1Loss(BaseLossFunction):
    """L1 Loss Function."""
    def __init__(self):
       self.loss_fn = nn.L1Loss()

    def compute(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, label)
