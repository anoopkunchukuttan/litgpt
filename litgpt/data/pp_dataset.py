# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class PretrainProcessedDataset(DataModule):
    """A generic data module for preprocessed streaming datasets.

    Provides training and validation streaming dataloaders that return batches of tokens.
    Works with any dataset that has been preprocessed into the litdata format with train/val splits.
    Supports multiple validation datasets for domain-specific or language-specific evaluation.
    """

    train_data_path: Union[str, Path] = None
    """The path to the train data directory
    which is the output of the preprocessing step done in advance (e.g., using prepare_pp_dataset.py).
    The path can also be a remote path (e.g., s3://)."""
    val_data_path: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None
    """The path(s) to validation data. Can be:
    - None: invalid
    - str/Path: Single validation dataset path
    - Dict[str, str/Path]: Multiple validation datasets with custom names
      Example: {'val_main': 'data/val_main', 'val_domain1': 'data/val_domain1'}
    """
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=4096)

    def __post_init__(self):
        super().__init__()
        
        # Handle validation paths
        if self.val_data_path is None:
            # Default: use 'val' subdirectory under data_path
            self.val_data_paths = {"val": str(self.data_path).rstrip("/") + "/val"}
            self._single_val_mode = True
        elif isinstance(self.val_data_path, dict):
            # Multiple validation datasets
            self.val_data_paths = {name: str(path).rstrip("/") for name, path in self.val_data_path.items()}
            self._single_val_mode = False
        else:
            # Single validation dataset (backward compatible)
            self.val_data_paths = {"val": str(self.val_data_path).rstrip("/")}
            self._single_val_mode = True
        
        self.required_paths = [self.train_data_path] + list(self.val_data_paths.values())

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        for path in self.required_paths:
            if not path.startswith("s3://") and not Path(path).is_dir():
                raise FileNotFoundError(
                    "The data path is expected to be a directory containing preprocessed data."
                    f" The directory {path} does not exist."
                    " Set it via `--data.train_data_path=...` or `--data.val_data_path=...`"
                )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_data = StreamingDataset(
            input_dir=self.train_data_path,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )

        train_dataloader = StreamingDataLoader(
            train_data, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        """
        Return a single validation dataloader (for backward compatibility).
        If multiple validation sets are configured, returns the first one.
        Use val_dataloaders() to get all validation sets.
        """
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        # Get the first validation set (maintains backward compatibility)
        first_val_name = list(self.val_data_paths.keys())[0]
        val_dataset = StreamingDataset(
            input_dir=self.val_data_paths[first_val_name],
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        val_dataloader = StreamingDataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader

    def val_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Return multiple validation dataloaders as a dictionary.
        This is the preferred method when multiple validation sets are configured.
        
        Returns:
            Dict[str, DataLoader]: Dictionary mapping validation set names to their dataloaders.
                                   For single validation sets, returns {'val': dataloader}.
        """
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        dataloaders = {}
        for val_name, val_path in self.val_data_paths.items():
            val_dataset = StreamingDataset(
                input_dir=val_path,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
            )
            dataloaders[val_name] = StreamingDataLoader(
                val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
            )
        
        return dataloaders
