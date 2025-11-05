import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple


class Visualizer:
    """
    Generic visualizer cho image tensors.

    Features:
    - Visualize bất kỳ dict structure nào (không fix keys)
    - Flexible columns và titles
    - Convert tensors to displayable images
    - Log images to WandB through logger
    """

    def __init__(self, logger=None):
        """
        Args:
            logger: WandbLogger instance (optional) để log images
        """
        self.logger = logger

    def visualize_samples(
        self,
        samples: List[Dict[str, torch.Tensor]],
        keys: Optional[List[str]] = None,
        titles: Optional[Dict[str, str]] = None,
        epoch: Optional[int] = None,
        caption: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Visualize samples với flexible structure.

        Args:
            samples: List of dicts, mỗi dict chứa tensors với arbitrary keys
            keys: List các keys cần visualize (theo order này). Nếu None, auto-detect từ sample đầu tiên
            titles: Dict mapping từ key -> display title. Nếu None, dùng key name làm title
            epoch: Epoch number (optional, để thêm vào caption)
            caption: Custom caption cho wandb log. Nếu None, auto-generate
            figsize: Figure size per row (width, height)

        Returns:
            matplotlib Figure object
        """
        if len(samples) == 0:
            return None

        # Auto-detect keys nếu không provide
        if keys is None:
            keys = list(samples[0].keys())

        # Auto-generate titles nếu không provide
        if titles is None:
            titles = {key: key.replace('_', ' ').title() for key in keys}

        num_samples = len(samples)
        num_cols = len(keys)

        # Create figure
        fig, axes = plt.subplots(
            num_samples,
            num_cols,
            figsize=(figsize[0], figsize[1] * num_samples)
        )

        # Handle single row case
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        # Handle single column case
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each sample
        for i, sample in enumerate(samples):
            for j, key in enumerate(keys):
                if key not in sample:
                    print(f"Warning: Key '{key}' not found in sample {i}")
                    continue

                # Convert tensor to image
                img = self.tensor_to_img(sample[key])

                # Plot
                axes[i, j].imshow(img)
                axes[i, j].set_title(titles[key])
                axes[i, j].axis('off')

        plt.tight_layout()

        # Log to wandb nếu có logger
        if self.logger:
            if caption is None:
                caption = f"epoch_{epoch}_results" if epoch is not None else "results"
            self.logger.wandb_image(fig, caption=caption)

        return fig

    def close_fig(self, fig):
        """Close matplotlib figure để free memory."""
        plt.close(fig)

    @staticmethod
    def tensor_to_img(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor sang displayable image.

        Args:
            tensor: Tensor [C, H, W] trong range [0, 1] hoặc [H, W] cho grayscale

        Returns:
            Numpy array [H, W, C] hoặc [H, W] trong range [0, 1]
        """
        # Handle different tensor shapes
        if tensor.dim() == 2:
            # Grayscale [H, W]
            img = tensor.numpy()
        elif tensor.dim() == 3:
            # RGB [C, H, W] -> [H, W, C]
            img = tensor.permute(1, 2, 0).numpy()
            # Squeeze nếu là single channel
            if img.shape[2] == 1:
                img = img.squeeze(2)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

        img = np.clip(img, 0, 1)
        return img
