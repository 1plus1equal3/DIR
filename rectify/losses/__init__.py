from .l1_loss import L1Loss
from .l2_loss import L2Loss
from .ce_loss import CrossEntropyLoss
from .bce_loss import BCELoss

LOSS_FUNCTIONS = {
    'l1': L1Loss,
    'l2': L2Loss,
    'ce': CrossEntropyLoss,
    'bce': BCELoss,
}