import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *
from .losses import LOSS_FUNCTIONS

class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.loss_fn = LOSS_FUNCTIONS[LOSS_FUNCTION]()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss_fn.compute(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()