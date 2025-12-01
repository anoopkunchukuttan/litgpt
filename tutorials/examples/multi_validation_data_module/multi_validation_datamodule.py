"""
Example: Custom DataModule with Multiple Validation Sets

This example demonstrates how to create a custom DataModule that supports
multiple validation sets for use with litgpt pretraining.

The changes to pretrain.py are backward compatible:
- If your DataModule has val_dataloader(), it returns a single validation set
- If your DataModule has val_dataloaders(), it returns multiple validation sets as a dict
"""

from pathlib import Path
from typing import Dict, Optional
from torch.utils.data import DataLoader, Dataset
from litgpt import Tokenizer
from litgpt.data import DataModule


class CustomDataModule(DataModule):
    """Example DataModule with multiple validation sets."""
    
    def __init__(
        self,
        train_data_path: Path,
        val_data_paths: Dict[str, Path],  # e.g., {"val_main": path1, "val_domain1": path2}
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 4,
        max_seq_length: int = 2048,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_paths = val_data_paths
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_datasets = {}
    
    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = None, max_seq_length: int = None):
        """Connect tokenizer and update parameters."""
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if batch_size is not None:
            self.batch_size = batch_size
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
    
    def prepare_data(self):
        """Download or prepare data (runs once on rank 0)."""
        pass
    
    def setup(self, stage: str = ""):
        """Setup datasets (runs on all ranks)."""
        # Load training dataset
        self.train_dataset = self._load_dataset(self.train_data_path)
        
        # Load multiple validation datasets
        for val_name, val_path in self.val_data_paths.items():
            self.val_datasets[val_name] = self._load_dataset(val_path)
    
    def _load_dataset(self, data_path: Path) -> Dataset:
        """Load a dataset from path (implement your loading logic here)."""
        # This is a placeholder - implement your actual dataset loading
        from litgpt.data import TextFiles
        # You would load your actual dataset here
        raise NotImplementedError("Implement your dataset loading logic")
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def val_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Return multiple validation dataloaders as a dictionary.
        
        This method is the key to multi-validation support!
        If this method exists, pretrain.py will use it instead of val_dataloader().
        """
        return {
            val_name: DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            for val_name, val_dataset in self.val_datasets.items()
        }


# Example YAML config for using this DataModule:
"""
model_name: pythia-14m
out_dir: out/pretrain/multi_val_example

# Use custom data module with multiple validation sets
data:
  class_path: examples.multi_validation_datamodule.CustomDataModule
  init_args:
    train_data_path: /path/to/train
    val_data_paths:
      val_main: /path/to/val_main       # Main validation set
      val_domain1: /path/to/val_domain1  # Domain-specific validation
      val_domain2: /path/to/val_domain2  # Another domain
    num_workers: 4

train:
  global_batch_size: 512
  micro_batch_size: 4
  max_tokens: 1000000000

eval:
  interval: 1000
  max_iters: 100
  initial_validation: true
  final_validation: true

tokenizer_dir: checkpoints/EleutherAI/pythia-14m

# WandB will log metrics as:
# - val_main/loss, val_main/ppl
# - val_domain1/loss, val_domain1/ppl
# - val_domain2/loss, val_domain2/ppl
logger_name: wandb
log:
  project: multi-validation-example
"""
