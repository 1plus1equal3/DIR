import torch
import torch.nn as nn
from .base import BaseLossFunction

class BCELoss(BaseLossFunction):
    """Binary Cross Entropy Loss Function."""
    def __init__(self):
        self.loss_fn = nn.BCELoss()

    def compute(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, label)