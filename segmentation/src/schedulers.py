from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings

def get_position_from_periods(iteration, cumulative_period):
    for i, period in enumerate(cumulative_period):
        if iteration < period:
            return i
    return len(cumulative_period) - 1

# --- Warmup Cosine Annealing Learning Rate Scheduler ---
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs, warmup_lr=None, eta_min=0, last_epoch=-1):
        """
        Warmup Cosine Annealing Learning Rate Scheduler.
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations.
            warmup_epochs (int): Number of warmup epochs.
            eta_min (float, optional): Minimum learning rate. Default: 0.
            last_epoch (int, optional): The index of the last epoch. Default: -1.
        """
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    #! NOTE: base_lrs is the snapshot of the initial learning rates of the optimizer
    def get_lr(self):
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )
        if self.warmup_lr is None:
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs
            ]
        if self.last_epoch == 0: # Return optimizer's initial learning rate
            return [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch <= self.warmup_epochs:
            # Linear warmup
            return [
                base_lr + (self.warmup_lr - base_lr) * self.last_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing after warmup
            return [
                self.eta_min + (self.warmup_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
                for _ in self.base_lrs
            ]

# --- Cosine Annealing Restart Warmup Scheduler ---
class CosineAnealingRestartWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, periods, restart_weights=(1,), eta_min=0, last_epoch=-1, warmup_target_lr=None, warmup_epochs=0):
        """
        Cosine Annealing Restart Warmup Scheduler.
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            periods (list): List of periods for each restart.
            restart_weights (tuple, optional): Weights for each restart period. Default: (1,).
            eta_min (float, optional): Minimum learning rate. Default: 0.
            last_epoch (int, optional): The index of the last epoch. Default: -1.
            warmup_target_lr (float, optional): Target learning rate for warmup. Default: None.
            warmup_epochs (int, optional): Number of warmup epochs. Default: 0.
        """
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        self.T_max = sum(periods)
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        self.warmup_target_lr = warmup_target_lr
        self.warmup_epochs = warmup_epochs
        super(CosineAnealingRestartWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                            self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        if self.warmup_target_lr is None:
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (
                    1 + math.cos(math.pi * (self.last_epoch - nearest_restart) / current_period)) * current_weight
                for base_lr in self.base_lrs
            ]
        else:
            if self.last_epoch <= self.warmup_epochs:
                #Start from initial lr and linearly increase to warmup_target_lr
                self.warmup_lr = [base_lr + (self.warmup_target_lr - base_lr) * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
                return self.warmup_lr
            else:
                if self.last_epoch < self.periods[0]:
                    # print(self.last_epoch, nearest_restart)
                    return [
                        self.eta_min + (base_lr - self.eta_min) * 0.5 * (
                            1 + math.cos(math.pi * (
                                (self.last_epoch - self.warmup_epochs - nearest_restart) / (current_period - self.warmup_epochs)))) * current_weight
                        for base_lr in self.warmup_lr
                    ]
                else:
                    return [
                        self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
                        (1 + math.cos(math.pi * (
                            (self.last_epoch - nearest_restart) / current_period)))
                        for base_lr in self.base_lrs
                    ]