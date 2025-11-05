import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Callable


class Evaluator:
    """
    Generic evaluator cho document rectification (hoặc bất kỳ image-to-image task nào).

    Features:
    - Flexible loss functions (không hard-code)
    - Batch và dataset evaluation
    - Collect samples cho visualization
    - Progress bar với tqdm
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        loss_functions: Optional[List[Dict]] = None
    ):
        """
        Args:
            model: Model để evaluate
            device: Device để run evaluation ('cuda' hoặc 'cpu')
            loss_functions: List of dicts chứa loss info. Mỗi dict có:
                - 'fn': Loss function object với compute() method
                - 'weight': Weight cho loss này
                - 'type': Tên của loss (để logging)
                Nếu None, chỉ collect outputs mà không compute loss
        """
        self.model = model.to(device)
        self.device = device
        self.loss_functions = loss_functions or []

    @torch.no_grad()
    def eval_step(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        return_output: bool = True
    ) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        """
        Single evaluation step.

        Args:
            input: Input tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            return_output: Có return output tensor không (để visualization)

        Returns:
            - Dict chứa loss values (empty nếu không có loss functions)
            - Output tensor (None nếu return_output=False)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input)

        # Compute losses
        loss_dict = {}
        if len(self.loss_functions) > 0:
            total_loss = 0

            for loss_info in self.loss_functions:
                loss_value = loss_info['fn'].compute(output, target)
                weighted_loss = loss_value * loss_info['weight']
                total_loss += weighted_loss

                loss_name = loss_info['type']
                loss_dict[loss_name] = loss_value.item()

            loss_dict['total'] = total_loss.item()

        return loss_dict, output if return_output else None

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        num_vis_samples: int = 4,
        sample_keys: Optional[List[str]] = None,
        desc: str = "Evaluation",
        collect_samples: bool = True
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate trên toàn bộ dataset.

        Args:
            dataloader: DataLoader để evaluate
            num_vis_samples: Số samples để collect cho visualization
            sample_keys: Keys để store trong samples dict. Default: ['input', 'target', 'output']
            desc: Description cho progress bar
            collect_samples: Có collect samples để visualize không

        Returns:
            - Dict chứa average losses (empty nếu không có loss functions)
            - List các samples để visualization (empty nếu collect_samples=False)
        """
        self.model.eval()

        # Default sample keys
        if sample_keys is None:
            sample_keys = ['input', 'target', 'output']

        # Initialize loss accumulator
        epoch_losses = {}
        if len(self.loss_functions) > 0:
            for loss_info in self.loss_functions:
                epoch_losses[loss_info['type']] = []
            epoch_losses['total'] = []

        # Store samples for visualization
        vis_samples = []

        pbar = tqdm(dataloader, desc=desc)

        for batch_idx, (input, target) in enumerate(pbar):
            input = input.to(self.device)
            target = target.to(self.device)

            # Evaluation step
            loss_dict, output = self.eval_step(
                input,
                target,
                return_output=collect_samples
            )

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

            # Collect samples for visualization (chỉ từ batch đầu tiên)
            if collect_samples and batch_idx == 0:
                num_samples = min(num_vis_samples, input.size(0))
                for i in range(num_samples):
                    sample = {}
                    # Dynamically add requested keys
                    if 'input' in sample_keys:
                        sample['input'] = input[i].cpu()
                    if 'target' in sample_keys:
                        sample['target'] = target[i].cpu()
                    if 'output' in sample_keys and output is not None:
                        sample['output'] = output[i].cpu()
                    vis_samples.append(sample)

            # Update progress bar
            if len(loss_dict) > 0 and 'total' in loss_dict:
                pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        # Compute epoch averages
        avg_losses = {}
        for key, values in epoch_losses.items():
            if len(values) > 0:
                avg_losses[key] = sum(values) / len(values)

        return avg_losses, vis_samples

    def set_loss_functions(self, loss_functions: List[Dict]):
        """
        Update loss functions.

        Args:
            loss_functions: List of dicts chứa loss info
        """
        self.loss_functions = loss_functions
