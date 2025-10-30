import wandb
from datetime import datetime

class WandbLogger:
    def __init__(self, project_name, api_key_path):
        self.project_name = project_name
        self.api_key_path = api_key_path
        # Initialize run
        self.init_run()

    def _get_wandb_key(self):
        try:
            with open(self.api_key_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Wandb API key file not found at {self.api_key_path}")

    def init_run(self):
        wandb.login(key=self._get_wandb_key())
        self.run = wandb.init(project=self.project_name, name=str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    def wandb_log_metric(self, dict):
        self.run.log(dict)

    def wandb_image(self, plot, caption: str):
        self.run.log({caption: wandb.Image(plot, caption)})