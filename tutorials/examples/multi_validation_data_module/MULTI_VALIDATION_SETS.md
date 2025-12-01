# Multiple Validation Sets Support

## Overview

LitGPT's pretraining now supports multiple validation sets while maintaining full backward compatibility with single validation sets. This allows you to track model performance across different domains, languages, or difficulty levels during training.

## Key Changes

### 1. **Backward Compatible Design**
- Existing code with single validation sets continues to work without any changes
- If your DataModule has `val_dataloader()`, it returns a single validation set
- If your DataModule has `val_dataloaders()`, it returns multiple validation sets as a dictionary

### 2. **Modified Functions**

#### `validate()` function
- Added `dataset_name` parameter for better logging
- Added empty batch handling (returns NaN instead of crashing)

#### `get_dataloaders()` function
- Returns `Union[DataLoader, Dict[str, DataLoader]]` for validation
- Automatically detects if DataModule supports multiple validation sets

#### `fit()` function
- Updated signature: `val_dataloader: Union[DataLoader, Dict[str, DataLoader]]`
- Normalizes single validation loader to dict internally for uniform handling
- Handles validation for all datasets during periodic checks
- Logs metrics with dataset-specific namespaces

#### `main()` function
- Handles Fabric setup for both single and multiple validation loaders

## Usage

### Option 1: Single Validation Set (Existing Behavior)

Your DataModule with `val_dataloader()` method:

```python
class MyDataModule(DataModule):
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, ...)
```

**Config:**
```yaml
data:
  class_path: litgpt.data.TextFiles
  init_args:
    train_data_path: /path/to/train
    val_data_path: /path/to/val
```

**Output logs:**
```
iter 1000: val loss 2.456
```

**WandB metrics:**
- `val/loss`
- `val/ppl`

---

### Option 2: Multiple Validation Sets (New Feature)

Your DataModule with `val_dataloaders()` method:

```python
class MyDataModule(DataModule):
    def val_dataloaders(self) -> Dict[str, DataLoader]:
        return {
            "val_main": DataLoader(self.val_main_dataset, ...),
            "val_domain1": DataLoader(self.val_domain1_dataset, ...),
            "val_domain2": DataLoader(self.val_domain2_dataset, ...),
        }
```

**Config:**
```yaml
data:
  class_path: my_module.MyDataModule
  init_args:
    train_data_path: /path/to/train
    val_data_paths:
      val_main: /path/to/val_main
      val_domain1: /path/to/val_domain1
      val_domain2: /path/to/val_domain2
```

**Output logs:**
```
iter 1000: val_main loss 2.456
iter 1000: val_domain1 loss 2.789
iter 1000: val_domain2 loss 3.012
Total validation time: 450.23 ms
```

**WandB metrics:**
- `val_main/loss`, `val_main/ppl`
- `val_domain1/loss`, `val_domain1/ppl`
- `val_domain2/loss`, `val_domain2/ppl`

## Example Implementation

See `examples/multi_validation_datamodule.py` for a complete example of a custom DataModule with multiple validation sets.

## Benefits

1. **Domain-Specific Tracking**: Monitor performance on different domains simultaneously
2. **Language-Specific Tracking**: Track multilingual model performance per language
3. **Difficulty-Based Tracking**: Evaluate on easy/medium/hard validation sets
4. **Comprehensive Evaluation**: All metrics logged to WandB/TensorBoard with clear namespacing

## Migration Guide

### If you have a single validation set:
No changes needed! Your existing code works as-is.

### If you want multiple validation sets:

1. Update your DataModule to implement `val_dataloaders()` instead of (or in addition to) `val_dataloader()`
2. Return a dictionary where keys are validation set names and values are DataLoaders
3. Update your config to specify multiple validation paths
4. Run training - metrics will be logged with dataset-specific prefixes

## Important Notes

- **Empty batches**: If a validation dataloader has no batches or `max_iters=0`, it will return NaN loss with a warning instead of crashing
- **Metric naming**: Validation metrics use the format `{dataset_name}/loss` and `{dataset_name}/ppl`
- **Validation time**: All validation sets are processed sequentially during each validation interval
- **Checkpoint saving**: Based on training steps, not validation performance (no automatic best model selection)

## Technical Details

### Type Signatures

```python
def validate(
    fabric: L.Fabric,
    model: nn.Module,
    val_dataloader: DataLoader,
    max_iters: int,
    verbose: bool = True,
    dataset_name: str = "val"  # New parameter
) -> torch.Tensor:
    ...

def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: Union[DataLoader, Dict[str, DataLoader]],  # Union type
    ...
) -> None:
    ...

def get_dataloaders(
    fabric: L.Fabric,
    data: DataModule,
    tokenizer: Tokenizer,
    train: TrainArgs,
    block_size: int
) -> Tuple[DataLoader, Union[DataLoader, Dict[str, DataLoader]]]:  # Union return type
    ...
```

### Internal Normalization

The `fit()` function normalizes the validation dataloader to a dictionary internally:

```python
# Normalize val_dataloader to dict for uniform handling
if not isinstance(val_dataloader, dict):
    val_dataloaders = {"val": val_dataloader} if val_dataloader is not None else {}
else:
    val_dataloaders = val_dataloader
```

This ensures all validation logic handles both single and multiple sets uniformly.
