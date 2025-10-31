import torch
import torch.nn as nn
from .base import BaseLossFunction

class CrossEntropyLoss(BaseLossFunction):
    """Cross Entropy Loss Function."""
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def compute(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, label)