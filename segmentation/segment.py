# %%
import glob
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# %% [markdown]
# # Data pipeline

# %%
def load_image(path, mode=cv2.IMREAD_COLOR):
    image = cv2.imread(path, mode)
    if mode == cv2.IMREAD_COLOR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def random_rotate(image, k=0):
    return torch.rot90(image, k, [1, 2])

class SegmentDataset(Dataset):
    """ 
    Dataset for image segmentation tasks.
    """
    def __init__(self, image_paths, mask_paths, resolution=(512, 512)):
        self.resolution = resolution
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        assert len(self.image_paths) == len(self.mask_paths), "Image and mask paths must be of the same length."
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution)
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = load_image(self.image_paths[idx])
        mask = load_image(self.mask_paths[idx], mode=cv2.IMREAD_GRAYSCALE)
        # Convert to tensor
        image = self.transform(image)
        mask = self.transform(mask)
        # Data augmentation: random rotation
        k = random.randint(0, 3)  # Rotate by 0, 90, 180, or 270 degrees
        image = torch.clip(random_rotate(image, k), 0, 1)
        mask = torch.clip(random_rotate(mask, k), 0, 1)
        return image, mask


# %%
# Root directory of the dataset
root_dir = '/home/user02/linhdang/DIR/Data'

# Load paths
image_paths = glob.glob(f'{root_dir}/part_*/warp/*')
mask_paths = glob.glob(f'{root_dir}/part_*/other/mask_*.png')
print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")

# Sort paths to ensure matching order
image_paths.sort()
mask_paths.sort()

# Divide into training and validation sets
split_ratio = 0.9
num_images = len(image_paths)
indices = list(range(num_images))
random.shuffle(indices)
split_index = int(num_images * split_ratio)
train_indices = indices[:split_index]
val_indices = indices[split_index:]
train_image_paths = [image_paths[i] for i in train_indices]
train_mask_paths = [mask_paths[i] for i in train_indices]
val_image_paths = [image_paths[i] for i in val_indices]
val_mask_paths = [mask_paths[i] for i in val_indices]

# Create dataset and dataloader
train_dataset = SegmentDataset(image_paths, mask_paths)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
print(f"Training dataset size: {len(train_dataset)}")
val_dataset = SegmentDataset(val_image_paths, val_mask_paths)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
print(f"Validation dataset size: {len(val_dataset)}")

# %%
images, masks = next(iter(train_dataloader))
print(f"Batch image shape: {images.shape}")  # (B, C, H, W)
print(f"Batch mask shape: {masks.shape}")    # (B, 1, H, W)

# %%
f, ax = plt.subplots(5, 2, figsize=(10, 5))
for i in range(5):
    ax[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
    ax[i, 0].axis('off')
    ax[i, 1].imshow(masks[i].squeeze().numpy(), cmap='gray')
    ax[i, 1].axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# # Model

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.seg import U2NETP
from torchsummary import summary

# %%
class U2NETP_DocSeg(nn.Module):
    def __init__(self, num_classes=1):
        super(U2NETP_DocSeg, self).__init__()
        self.u2netp = U2NETP()

    def forward(self, x):
        mask, *_ = self.u2netp(x)
        return mask
    
model = U2NETP_DocSeg(num_classes=1).cuda()
summary(model, (3, 512, 512))

# %% [markdown]
# # Train preparation

# %%
from src.lovasz_losses import lovasz_softmax

# Initialize loss function
mask_loss_fn = nn.BCEWithLogitsLoss()
lovasz_fn = lovasz_softmax

# %%
import torchmetrics
from torchmetrics.segmentation import MeanIoU, DiceScore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training and evaluation metrics
train_metrics = {
    'train_iou': MeanIoU(num_classes=2).to(device),
    'train_dice': DiceScore(num_classes=2).to(device),
    'train_mask_loss': torchmetrics.MeanMetric().to(device),
}

eval_metrics = {
    'eval_iou': MeanIoU(num_classes=2).to(device),
    'eval_dice': DiceScore(num_classes=2).to(device),
    'eval_mask_loss': torchmetrics.MeanMetric().to(device),
}   

def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset()

# %%
from src.schedulers import WarmupCosineAnnealingLR

scheduler_config = {
    'T_max': 200, # Number of epochs
    'warmup_epochs': 10, # Warmup epochs
    'warmup_lr': 2e-4,
    'eta_min': 1e-5
}

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
scheduler = WarmupCosineAnnealingLR(
    optimizer,
    **scheduler_config
)

# %%
from src.checkpoint import CheckpointManager
from datetime import datetime

ckpt_dir = f'checkpoints_{datetime.now().strftime("%y_%m_%d")}'
os.makedirs(ckpt_dir, exist_ok=True)

train_config = {
    'initial_epoch': 1,
    'num_epochs': 250,
    'step_per_epoch': 250,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'eval_epochs': 5,  # Evaluate every 5 epochs
    'visualize_interval': 5,  # Visualize every 5 epochs
    'save_interval': 5,  # Save model every 5 epochs
    'model_name': 'u2netp_docseg',
    'ckpt_dir': ckpt_dir,
    'max_ckpt_num': 10,  # Keep the last 10 checkpoints
}

checkpoint_manager = CheckpointManager(
    train_config['model_name'],
    train_config['ckpt_dir'],
    train_config['max_ckpt_num'],
)

ckpt_dict = {
    'model': model,
    'optimizer': optimizer,
    'schedule': scheduler,
}

# Restore from checkpoint if exists
latest_ckpt = None
if latest_ckpt is None: 
    latest_ckpt = checkpoint_manager.get_latest_checkpoint()
if latest_ckpt:
    last_epoch = checkpoint_manager.load_checkpoint(latest_ckpt, ckpt_dict)
    train_config['initial_epoch'] = last_epoch

# %%
from src.logger import WandbLogger

wandb_logger = WandbLogger(
    project_name='DocSeg',
    api_key_path='./wandb.txt'
)

# %% [markdown]
# # Training

# %%
def train_step(data, device):
    # Parse data
    images, masks = data
    # Move to device
    images = images.to(device)
    masks = masks.to(device)

    # Forward pass
    pred_mask = model(images)
    # Compute losses
    mask_loss = mask_loss_fn(pred_mask, masks) # Binary cross-entropy loss for masks
    # lovasz_loss = lovasz_fn(pred_mask, masks) # Lovasz loss for masks
    total_loss = mask_loss # * 0.8 + lovasz_loss * 0.2

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()    

    # Update metrics
    pred_mask = (pred_mask > 0.5).int()
    with torch.no_grad():
        train_metrics['train_iou'].update(pred_mask, masks.int())
        train_metrics['train_dice'].update(pred_mask, masks.int())
        train_metrics['train_mask_loss'].update(mask_loss)

def eval_step(data, device):
    # Parse data
    images, masks = data
    # Move to device
    images = images.to(device)
    masks = masks.to(device)

    # Forward pass
    pred_mask = model(images)
    mask_loss = mask_loss_fn(pred_mask, masks) # Binary cross-entropy loss for masks

    pred_mask = (pred_mask > 0.5).int()
    # Update metrics
    with torch.no_grad():
        eval_metrics['eval_iou'].update(pred_mask, masks.int())
        eval_metrics['eval_dice'].update(pred_mask, masks.int())
        eval_metrics['eval_mask_loss'].update(mask_loss)

# %%
def visualize(ds_val, sample_num, caption='Visualization', log_wandb=False):
    batch = next(iter(ds_val))
    images, masks = batch
    images = images[:sample_num]
    masks = masks[:sample_num]

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        pred_mask = model(images)
    pred_mask = (pred_mask > 0.5).float()

    # Visualization
    col_num = 3  # 2 for image and label, plus number of masks
    fig, ax = plt.subplots(
        sample_num, col_num, figsize=(5 * col_num, 5 * sample_num),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.75},
    )
    fig.suptitle('IMG / GT / PRED', fontsize=24, y=1.01)
    fig.subplots_adjust(top=0.98)
    for i in range(sample_num):
        ax[i, 0].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        ax[i, 0].set_title('Input Image')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(masks[i].cpu().permute(1, 2, 0).numpy(), cmap='gray')
        ax[i, 1].set_title('Ground Truth')
        ax[i, 1].axis('off')

        ax[i, 2].imshow(pred_mask[i].cpu().permute(1, 2, 0).numpy(), cmap='gray')
        ax[i, 2].set_title('Predicted Mask')
        ax[i, 2].axis('off')
    if log_wandb:
        wandb_logger.wandb_image(fig, caption)
    plt.show()

# Test visualization
visualize(val_dataloader, sample_num=4, caption='Train Batch Visualization', log_wandb=False)

# %%
def logging(step, metric_values):
    total_steps = train_config['step_per_epoch']
    metric_str = f"\rStep {step+1}/{total_steps} - " + " - ".join(
        f"{name}: {value:.4f}" for name, value in metric_values.items()
    )
    sys.stdout.write(metric_str)
    sys.stdout.flush()
    wandb_logger.wandb_log_metric(metric_values)

def eval(ds_val, device):
    # Set model to evaluation mode
    model.eval()
    reset_metrics(eval_metrics)
    for batch in ds_val:
        eval_step(batch, device)
    # Compute final metrics
    metric_values = {
        'eval_iou': eval_metrics['eval_iou'].compute(),
        'eval_dice': eval_metrics['eval_dice'].compute(),
        'eval_mask_loss': eval_metrics['eval_mask_loss'].compute(),
    }
    print(f"Evaluation Metrics: {metric_values}")
    return metric_values

# Training loop
def train(initial_epoch, ds_train, ds_val, device):
    # Set model to training mode
    model.train()
    for epoch in range(initial_epoch, train_config['num_epochs'] + 1):
        try:
            reset_metrics(train_metrics)
            print(f'Epoch {epoch}/{train_config["num_epochs"]}')
            for step, batch in enumerate(ds_train):
                if step >= train_config['step_per_epoch']: break # Limit to steps per epoch
                train_step(batch, device)
                # Log metrics
                metric_values = {
                    'train_iou': train_metrics['train_iou'].compute(),
                    'train_dice': train_metrics['train_dice'].compute(),
                    'train_mask_loss': train_metrics['train_mask_loss'].compute(),
                }
                logging(step, metric_values)
            # Step the scheduler
            scheduler.step()
            # Evaluate model
            if epoch % train_config['eval_epochs'] == 0:
                eval_metrics = eval(ds_val, device)
                print(f"\nEpoch {epoch+1} Evaluation Metrics: {eval_metrics}")
                # Log evaluation metrics to wandb
                wandb_logger.wandb_log_metric(eval_metrics)
            # Visualize and save model
            if epoch % train_config['visualize_interval'] == 0:
                visualize(ds_val, sample_num=4, caption=f'Epoch {epoch+1} Visualization', log_wandb=True)
            # Save model checkpoint
            if epoch % train_config['save_interval'] == 0:
                checkpoint_manager.save_checkpoint(ckpt_dict, epoch)
                print(f"\nCheckpoint saved for epoch {epoch+1}")
            print()
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            checkpoint_manager.save_checkpoint(ckpt_dict, epoch)
            print("Checkpoint saved. Exiting training loop.")
            return


# %%
#! Start training
device = train_config['device']
model.to(device)
train(
    initial_epoch=train_config['initial_epoch'],
    ds_train=train_dataloader,
    ds_val=val_dataloader,
    device=device
)


