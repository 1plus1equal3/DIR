import torch
import torch.nn as nn
from .base import BaseLossFunction

class L2Loss(BaseLossFunction):
    """L2 Loss Function."""
    def __init__(self):
       self.loss_fn = nn.MSELoss()

    def compute(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, label)
