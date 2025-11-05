import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

from .losses import LOSS_FUNCTIONS
from .visualizer import Visualizer
from .evaluator import Evaluator

import sys
sys.path.append('./segmentation/src')
from segmentation.src.logger import WandbLogger
from segmentation.src.checkpoint import CheckpointManager


# Default config values
DEFAULT_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'log_every_n_steps': 10,
    'num_vis_samples': 4,
    'val_every_n_epochs': 1,
    'visualize_every_n_epochs': 5,
    'save_every_n_epochs': 5,
    'max_checkpoints': 5,
    'wandb_project': 'document-rectification',
    'wandb_api_key_path': './wandb_key.txt',
    'checkpoint_dir': './checkpoints',
    'model_name': 'rectify_model',
}


class Trainer:
    """
    Trainer class cho document rectification task.

    Features:
    - Multi-loss function support với weights
    - Training và evaluation loops
    - WandB logging (metrics + images)
    - Checkpoint management với resume capability
    - Visualization của results
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        logger: WandbLogger,
        checkpoint_manager: CheckpointManager
    ):
        """
        Args:
            model: Model để train (CustomUNet)
            config: Config dictionary chứa training config và loss functions
            logger: WandbLogger instance để log metrics và images
            checkpoint_manager: CheckpointManager instance để save/load checkpoints
        """
        # Merge với default config
        self.config = {**DEFAULT_CONFIG, **config}

        # Extract training config
        training_config = self.config.get('training', {})

        self.model = model.to(self.config['device'])
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

        # Initialize multiple loss functions với weights
        self.loss_functions = []
        loss_function_configs = training_config.get('loss_function', [])
        for loss_config in loss_function_configs:
            loss_cls = LOSS_FUNCTIONS[loss_config['type']]
            loss_fn = loss_cls(**loss_config.get('params', {}))
            weight = loss_config.get('weight', 1.0)
            loss_type = loss_config['type']
            self.loss_functions.append({
                'fn': loss_fn,
                'weight': weight,
                'type': loss_type
            })

        # Initialize optimizer
        lr = self.config.get('learning_rate', 1e-4)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        # Initialize visualizer và evaluator
        self.visualizer = Visualizer(logger=self.logger)
        self.evaluator = Evaluator(
            model=self.model,
            device=self.config['device'],
            loss_functions=self.loss_functions
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.

        Args:
            input: Input tensor (warped images) [B, C, H, W]
            target: Target tensor (flat images) [B, C, H, W]

        Returns:
            Dict chứa loss values cho mỗi loss function
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(input)

        # Compute weighted sum of losses
        total_loss = 0
        loss_dict = {}

        for loss_info in self.loss_functions:
            loss_value = loss_info['fn'].compute(output, target)
            weighted_loss = loss_value * loss_info['weight']
            total_loss += weighted_loss

            # Store individual loss values
            loss_name = f"train/{loss_info['type']}_loss"
            loss_dict[loss_name] = loss_value.item()

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        loss_dict['train/total_loss'] = total_loss.item()
        return loss_dict


    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train cho một epoch.

        Args:
            train_loader: Training dataloader

        Returns:
            Dict chứa average losses cho epoch này
        """
        epoch_losses = {
            f"train/{loss_info['type']}_loss": []
            for loss_info in self.loss_functions
        }
        epoch_losses['train/total_loss'] = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch}/{self.config['num_epochs']} [Train]"
        )

        for batch_idx, (input, target) in enumerate(pbar):
            input = input.to(self.config['device'])
            target = target.to(self.config['device'])

            # Training step
            loss_dict = self.train_step(input, target)

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['train/total_loss']:.4f}"
            })

            # Log to wandb
            if self.global_step % self.config['log_every_n_steps'] == 0:
                self.logger.wandb_log_metric(loss_dict)

            self.global_step += 1

        # Compute epoch averages
        avg_losses = {
            key: np.mean(values)
            for key, values in epoch_losses.items()
        }
        return avg_losses



    def save_checkpoint(self):
        """Save checkpoint với current state."""
        ckpt_dict = {
            'model': self.model,
            'optimizer': self.optimizer,
        }
        self.checkpoint_manager.save_checkpoint(ckpt_dict, self.current_epoch)

    def load_checkpoint(self, ckpt_path: str):
        """
        Load checkpoint từ path.

        Args:
            ckpt_path: Path đến checkpoint file
        """
        ckpt_dict = {
            'model': self.model,
            'optimizer': self.optimizer,
        }
        self.current_epoch = self.checkpoint_manager.load_checkpoint(
            ckpt_path,
            ckpt_dict
        )
        self.global_step = self.current_epoch * 1000  # Estimate global step

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """
        Main training loop.

        Args:
            train_loader: Training dataloader (được provide bởi người làm data)
            val_loader: Validation dataloader (optional)
            resume_from: Path đến checkpoint để resume (optional)
        """
        # Load checkpoint nếu có
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed from epoch {self.current_epoch}")

        # Print training info
        print("=" * 60)
        print(f"Starting training from epoch {self.current_epoch}")
        print(f"Training on device: {self.config['device']}")
        print(f"Total epochs: {self.config['num_epochs']}")
        print(f"Loss functions: {[loss['type'] for loss in self.loss_functions]}")
        print(f"Train dataset size: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Val dataset size: {len(val_loader.dataset)}")
        print("=" * 60)

        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch

            # ========== Training ==========
            train_losses = self.train_epoch(train_loader)
            print(f"\nEpoch {epoch} - Train Loss: {train_losses['train/total_loss']:.4f}")

            # Log epoch average metrics (rename để tách biệt với step-level logging)
            epoch_train_metrics = {
                key.replace('train/', 'train_epoch/'): value
                for key, value in train_losses.items()
            }
            self.logger.wandb_log_metric(epoch_train_metrics)

            # ========== Validation ==========
            if val_loader and (epoch % self.config['val_every_n_epochs'] == 0):
                val_losses, vis_samples = self.evaluator.evaluate(
                    val_loader,
                    num_vis_samples=self.config['num_vis_samples'],
                    desc="Validation"
                )
                print(f"Epoch {epoch} - Val Loss: {val_losses['total']:.4f}")

                # Log validation epoch metrics với prefix
                epoch_val_metrics = {
                    f"val_epoch/{key}": value
                    for key, value in val_losses.items()
                }
                self.logger.wandb_log_metric(epoch_val_metrics)

                # Visualize results
                if epoch % self.config['visualize_every_n_epochs'] == 0:
                    fig = self.visualizer.visualize_samples(
                        samples=vis_samples,
                        keys=['input', 'output', 'target'],
                        titles={
                            'input': 'Input (Warped)',
                            'output': 'Output (Predicted)',
                            'target': 'Target (Ground Truth)'
                        },
                        epoch=epoch
                    )
                    self.visualizer.close_fig(fig)

            # ========== Save checkpoint ==========
            if epoch % self.config['save_every_n_epochs'] == 0 or epoch == self.config['num_epochs'] - 1:
                self.save_checkpoint()

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)


# ========== Helper function để khởi tạo trainer ==========
def create_trainer(
    model: nn.Module,
    config: Dict
) -> Trainer:
    """
    Helper function để khởi tạo Trainer với logger và checkpoint manager.

    Args:
        model: Model để train
        config: Config dictionary chứa training settings, loss functions, và các thông số khác

    Returns:
        Trainer instance đã được khởi tạo
    """
    # Merge với default config
    full_config = {**DEFAULT_CONFIG, **config}

    # Initialize logger
    logger = WandbLogger(
        project_name=full_config['wandb_project'],
        api_key_path=full_config['wandb_api_key_path']
    )

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        model_name=full_config['model_name'],
        save_dir=full_config['checkpoint_dir'],
        max_checkpoints=full_config['max_checkpoints']
    )

    # Initialize trainer
    trainer = Trainer(model, config, logger, checkpoint_manager)

    return trainer
