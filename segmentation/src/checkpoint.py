import os
import torch
from datetime import datetime
cur_date = str(datetime.now().strftime("%Y-%m-%d"))

class CheckpointManager():
    def __init__(self, model_name, save_dir, max_checkpoints=10):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, ckpt_dict, epoch):
        cur_date = str(datetime.now().strftime("%Y-%m-%d"))
        checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch}_date_{cur_date}.pth")
        if len(os.listdir(self.save_dir)) >= self.max_checkpoints:
            # Remove the oldest checkpoint if max checkpoints reached
            oldest_checkpoint = sorted(os.listdir(self.save_dir), key=lambda x: os.path.getctime(os.path.join(self.save_dir, x)))[0]
            os.remove(os.path.join(self.save_dir, oldest_checkpoint))
            print(f"Removed oldest checkpoint: {oldest_checkpoint}")
        # Save the checkpoint
        save_dict = {
            'epoch': epoch,
        }
        for key, value in ckpt_dict.items():
            save_dict.update({
                f"{key}_state_dict": value.state_dict() if hasattr(value, 'state_dict') else value
            })
        torch.save(save_dict, checkpoint_path)
        print(f"Epoch {epoch} - Checkpoint saved at {checkpoint_path}")

    def get_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith(self.model_name) and f.endswith('.pth')]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(self.save_dir, x)))
        return os.path.join(self.save_dir, latest_checkpoint)
    
    def load_checkpoint(self, ckpt_path, ckpt_dict):
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} does not exist.")
            return 0
        checkpoint = torch.load(ckpt_path)
        for key, value in ckpt_dict.items():
            if hasattr(value, 'load_state_dict'):
                value.load_state_dict(checkpoint[f"{key}_state_dict"])
            else:
                ckpt_dict[key] = checkpoint[f"{key}_state_dict"]
        print(f"Checkpoint loaded from {ckpt_path}")
        return checkpoint['epoch']