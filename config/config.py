from pathlib import Path
from typing import Optional

class Config:

    device='cuda'

    learning_rate:float = 2e-4
    batch_size:int=8
    num_epochs:int = 250

    lambda_cycle:int = 10
    
    model_folder: str = "weights"

    @staticmethod
    def latest_weights_file_path() -> Optional[str]:
        checkpoints = list(Path(Config.model_folder).glob('*_checkpoint_*.pth'))
        if checkpoints:
            # Extract the epoch number from each checkpoint file name and find the maximum
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            return str(latest_checkpoint)
        
        return None