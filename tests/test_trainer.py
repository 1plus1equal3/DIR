# -*- coding: utf-8 -*-
"""
Test script for Trainer class with tiny model and dataset.
"""
import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import from rectify package
from rectify.train import create_trainer, Trainer


# ========== Tiny Model ==========
class TinyModel(nn.Module):
    """
    Super tiny model to test trainer.
    Input: [B, 3, 32, 32]
    Output: [B, 3, 32, 32]
    """
    def __init__(self):
        super(TinyModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 32x32x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16x8

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8x16
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 8x8x16
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x16x16

            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # 16x16x8
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32x8

            nn.Conv2d(8, 3, kernel_size=3, padding=1),  # 32x32x3
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ========== Tiny Dataset ==========
class TinyDataset(Dataset):
    """
    Fake dataset with random images to test trainer.

    Args:
        num_samples: Number of samples
        image_size: Image size (height, width)
    """
    def __init__(self, num_samples=50, image_size=(32, 32)):
        self.num_samples = num_samples
        self.image_size = image_size

        # Generate random data
        np.random.seed(42)  # For reproducibility

        # Input: warped images (random noise)
        self.inputs = torch.rand(num_samples, 3, *image_size)

        # Target: flat images (slightly modified inputs for learning task)
        self.targets = torch.rand(num_samples, 3, *image_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ========== Test Function ==========
def test_trainer():
    """
    Test trainer with tiny model and dataset.
    """
    print("=" * 60)
    print("Testing Trainer with Tiny Model and Dataset")
    print("=" * 60)

    # ========== Config for test ==========
    # Load config from rectify/config/simple.py and override for quick testing
    from rectify.config import simple

    test_config = {
        'training': simple.training,
        'num_epochs': 3,
        'learning_rate': 1e-3,
        'log_every_n_steps': 5,
        'save_every_n_epochs': 2,
        'val_every_n_epochs': 1,
        'visualize_every_n_epochs': 1,
        'num_vis_samples': 2,
        'checkpoint_dir': './test_checkpoints',
        'model_name': 'tiny_test_model',
        'wandb_project': 'test_rectify',
    }

    print("\nTest configs:")
    print(f"  - Epochs: {test_config['num_epochs']}")
    print(f"  - Learning rate: {test_config['learning_rate']}")
    print("  - Image size: 32x32")
    print("  - Model params: ~few thousands")

    # ========== Create datasets ==========
    print("\nCreating datasets...")
    train_dataset = TinyDataset(num_samples=50, image_size=(32, 32))
    val_dataset = TinyDataset(num_samples=20, image_size=(32, 32))

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # 0 for simple testing
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    print("  - Batch size: 8")

    # ========== Create model ==========
    print("\nCreating tiny model...")
    model = TinyModel()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total params: {total_params:,}")
    print(f"  - Trainable params: {trainable_params:,}")

    # ========== Create trainer ==========
    print("\nCreating trainer...")
    try:
        trainer = create_trainer(
            model=model,
            config=test_config
        )
        print("  SUCCESS: Trainer created successfully!")
    except Exception as e:
        print(f"  ERROR: Failed to create trainer: {e}")
        print("\nNote: WandB login might fail if API key is missing.")
        print("      This is expected for testing. Skipping WandB logging...")
        import traceback
        traceback.print_exc()
        return

    # ========== Start training ==========
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    try:
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from=None
        )
        print("\nSUCCESS: Training completed successfully!")
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()

    # ========== Test inference ==========
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        sample_input = torch.rand(1, 3, 32, 32)
        sample_output = model(sample_input)
        print(f"  - Input shape: {sample_input.shape}")
        print(f"  - Output shape: {sample_output.shape}")
        print(f"  - Output range: [{sample_output.min():.3f}, {sample_output.max():.3f}]")
        print("  SUCCESS: Inference working!")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


# ========== Alternative test without WandB ==========
def test_trainer_no_wandb():
    """
    Test trainer WITHOUT WandB (for quick testing without API key).
    """
    print("=" * 60)
    print("Testing Trainer WITHOUT WandB")
    print("=" * 60)

    # Import locally to avoid circular import
    from rectify.config import simple
    from segmentation.src.checkpoint import CheckpointManager

    # ========== Mock Logger ==========
    class MockLogger:
        """Mock logger that doesn't require WandB."""
        def __init__(self):
            print("  Using mock logger (no WandB)")

        def wandb_log_metric(self, metrics_dict):
            # Print metrics instead of logging to WandB
            step = metrics_dict.get('step', metrics_dict.get('epoch', '?'))
            if 'step' in metrics_dict:
                loss = metrics_dict.get('train/total_loss', '?')
                if isinstance(loss, float):
                    print(f"    [Step {step}] Loss: {loss:.4f}")

        def wandb_image(self, fig, caption):
            print(f"    [Image] {caption}")

    # ========== Create components ==========
    print("\nCreating components...")

    # Config
    test_config = {
        'training': simple.training,
        'num_epochs': 2,
        'learning_rate': 1e-3,
        'log_every_n_steps': 5,
        'save_every_n_epochs': 1,
        'val_every_n_epochs': 1,
        'visualize_every_n_epochs': 1,
        'num_vis_samples': 2,
    }

    # Model
    model = TinyModel()
    print(f"  SUCCESS: Model created ({sum(p.numel() for p in model.parameters()):,} params)")

    # Logger
    logger = MockLogger()

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        model_name="tiny_test_model_no_wandb",
        save_dir="./test_checkpoints_no_wandb",
        max_checkpoints=5
    )
    print("  SUCCESS: Checkpoint manager created")

    # Trainer
    trainer = Trainer(model, test_config, logger, checkpoint_manager)
    print("  SUCCESS: Trainer created")

    # Datasets
    train_dataset = TinyDataset(num_samples=32, image_size=(32, 32))
    val_dataset = TinyDataset(num_samples=16, image_size=(32, 32))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    print(f"  SUCCESS: Datasets created (train: {len(train_dataset)}, val: {len(val_dataset)})")

    # ========== Train ==========
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    try:
        trainer.fit(train_loader, val_loader)
        print("\nSUCCESS: Training completed successfully!")
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    print("\nChoose test mode:")
    print("  1. Test with WandB (requires API key)")
    print("  2. Test without WandB (mock logger)")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1 or 2, default=2): ").strip() or "2"

    if choice == "1":
        test_trainer()
    else:
        test_trainer_no_wandb()
