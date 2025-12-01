# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import glob
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class TextFilesMultiVal(DataModule):
    """TextFile data module with multiple validation sets support.

    Reads in text data from plaintext files contained in a data folder
    and provides training and multiple validation dataloaders that return batches of tokens.
    Every sample is set to a fixed length.
    
    This module supports:
    - Single validation set (backward compatible with TextFiles)
    - Multiple named validation sets for domain-specific or language-specific evaluation
    - Auto-split from training data if no validation path provided
    """

    train_data_path: Path
    """The path to the data directory used for training that contains .txt files"""
    val_data_path: Optional[Union[Path, Dict[str, Path]]] = None
    """The path to the data directory used for validation that contains .txt files.
    Can be:
    - None: Splits off first .txt file from training data for validation
    - Path: Single validation set (backward compatible)
    - Dict[str, Path]: Multiple validation sets with custom names
      Example: {'val_main': Path('/data/val'), 'val_domain1': Path('/data/domain1')}
    """
    seed: int = 42
    """The seed to use for shuffling the dataset."""
    num_workers: int = 4
    """The number of workers to use for data loading."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.out_path_train = self.train_data_path / "train"
        
        # Handle both single and multiple validation paths
        if self.val_data_path is None:
            # Use default validation split from training data
            self.out_path_val = {"val": self.train_data_path / "val"}
            self._single_val_mode = True
        elif isinstance(self.val_data_path, dict):
            # Multiple validation sets
            self.out_path_val = {name: Path(path) / "val" for name, path in self.val_data_path.items()}
            self._single_val_mode = False
        else:
            # Single validation set (backward compatible)
            self.out_path_val = {"val": Path(self.val_data_path) / "val"}
            self._single_val_mode = True

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from litdata import optimize
        from litdata.streaming import TokensLoader

        train_files = sorted(glob.glob(str(self.train_data_path / "*.txt")))
        assert len(train_files) > 0, f"No .txt files found in train data {train_files}"

        # Determine validation files for each validation set
        val_files_dict = {}
        
        if self.val_data_path is None:
            # Split from training data - use first file for validation
            assert len(train_files) > 1, f"Expected at least two .txt files in {train_files}"
            val_file, *train_files = train_files
            val_files_dict["val"] = [val_file]
        elif isinstance(self.val_data_path, dict):
            # Multiple validation sets
            for val_name, val_path in self.val_data_path.items():
                val_path = Path(val_path)
                files = sorted(glob.glob(str(val_path / "*.txt")))
                assert len(files) > 0, f"No .txt files found in validation data '{val_name}': {val_path}"
                val_files_dict[val_name] = files
        else:
            # Single validation set
            val_path = Path(self.val_data_path)
            files = sorted(glob.glob(str(val_path / "*.txt")))
            assert len(files) > 0, f"No .txt files found in validation data {val_path}"
            val_files_dict["val"] = files

        # Process training data
        num_workers = os.cpu_count() - 1
        use_workers = min(num_workers, len(train_files))
        if not Path(self.out_path_train).is_dir():
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=train_files,
                output_dir=str(self.out_path_train),
                num_workers=use_workers,
                chunk_bytes="50MB",
                item_loader=TokensLoader(block_size=self.max_seq_length),
            )
        else:
            print(
                f"\nWarning: Preprocessed training data found in {self.out_path_train}."
                " For efficiency, reprocessing is skipped. If your text input has changed since"
                " the last `litgpt pretrain` command, remove the preprocessed file(s) to trigger"
                f" reprocessing: `rm -rf {self.out_path_train}`\n"
            )
        
        # Process each validation set
        for val_name, val_files in val_files_dict.items():
            out_path = self.out_path_val[val_name]
            use_workers = min(num_workers, len(val_files))
            if not Path(out_path).is_dir():
                validate_tokenizer(self.tokenizer)
                print(f"Processing validation set '{val_name}' with {len(val_files)} file(s)...")
                optimize(
                    fn=partial(tokenize, tokenizer=self.tokenizer),
                    inputs=val_files,
                    output_dir=str(out_path),
                    num_workers=use_workers,
                    chunk_bytes="50MB",
                    item_loader=TokensLoader(block_size=self.max_seq_length),
                )
            else:
                print(
                    f"\nWarning: Preprocessed validation data found for '{val_name}' in {out_path}."
                    " For efficiency, reprocessing is skipped. If your text input has changed since"
                    " the last `litgpt pretrain` command, remove the preprocessed file(s) to trigger"
                    f" reprocessing: `rm -rf {out_path}`\n"
                )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=str(self.out_path_train),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )

        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
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
        first_val_name = list(self.out_path_val.keys())[0]
        val_dataset = StreamingDataset(
            input_dir=str(self.out_path_val[first_val_name]),
            item_loader=TokensLoader(block_size=self.max_seq_length),
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
        for val_name, out_path in self.out_path_val.items():
            val_dataset = StreamingDataset(
                input_dir=str(out_path),
                item_loader=TokensLoader(block_size=self.max_seq_length),
                shuffle=True,
            )
            dataloaders[val_name] = StreamingDataLoader(
                val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
            )
        
        return dataloaders


def tokenize(filename: str, tokenizer: Tokenizer):
    with open(filename, encoding="utf-8") as file:
        text = file.read()
    text = text.strip()
    yield tokenizer.encode(text, bos=True, eos=False)


def validate_tokenizer(tokenizer: Tokenizer) -> None:
    if tokenizer is None:
        raise ValueError(
            "Tokenizer is None. If you are using this data module via `litgpt pretrain`, "
            "please provide a valid `--tokenizer_dir` path."
        )
