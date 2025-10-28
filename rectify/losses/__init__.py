from .l1_loss import L1Loss
from .l2_loss import L2Loss

LOSS_FUNCTIONS = {
    'l1': L1Loss,
    'l2': L2Loss,
}