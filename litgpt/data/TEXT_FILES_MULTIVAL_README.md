# TextFilesMultiVal - Multiple Validation Sets Support

## Overview

`TextFilesMultiVal` is an enhanced version of the `TextFiles` data module that supports multiple validation sets for comprehensive model evaluation during pretraining. It maintains full backward compatibility with the original `TextFiles` module.

## Features

✅ **Backward Compatible** - Works as a drop-in replacement for `TextFiles`  
✅ **Multiple Validation Sets** - Track performance across different domains, languages, or difficulty levels  
✅ **Auto-Split Support** - Automatically split validation from training data  
✅ **Named Validation Sets** - Custom names for each validation set  
✅ **Comprehensive Logging** - Each validation set gets its own metrics in WandB/TensorBoard

## Usage

### Option 1: Single Validation Set (Backward Compatible)

```yaml
data:
  class_path: litgpt.data.TextFilesMultiVal
  init_args:
    train_data_path: /path/to/train
    val_data_path: /path/to/val
    num_workers: 4
```

**Behavior:**
- Works exactly like the original `TextFiles` module
- Single validation set named "val"
- Metrics logged as `val/loss` and `val/ppl`

### Option 2: Multiple Validation Sets

```yaml
data:
  class_path: litgpt.data.TextFilesMultiVal
  init_args:
    train_data_path: /path/to/train
    val_data_path:
      val_main: /path/to/val_main
      val_domain1: /path/to/val_domain1
      val_domain2: /path/to/val_domain2
      val_lang_en: /path/to/val_english
      val_lang_es: /path/to/val_spanish
    num_workers: 4
```

**Behavior:**
- Multiple validation sets with custom names
- Each validation set is evaluated separately
- Metrics logged with namespace: `{name}/loss` and `{name}/ppl`
- Example: `val_main/loss`, `val_domain1/ppl`, etc.

### Option 3: Auto-Split from Training Data

```yaml
data:
  class_path: litgpt.data.TextFilesMultiVal
  init_args:
    train_data_path: /path/to/train  # Must contain at least 2 .txt files
    # val_data_path omitted
    num_workers: 4
```

**Behavior:**
- First `.txt` file (alphabetically) becomes validation set
- All remaining files become training set
- Validation set named "val"
- Useful for quick experiments

## Directory Structure

Each validation path should contain `.txt` files:

```
/path/to/train/
  ├── file1.txt
  ├── file2.txt
  └── file3.txt

/path/to/val_main/
  └── validation.txt

/path/to/val_domain1/
  ├── domain1_val1.txt
  └── domain1_val2.txt

/path/to/val_domain2/
  └── domain2_val.txt
```

## Example Output

### Console Logs (Multiple Validation Sets)

```
Processing validation set 'val_main' with 1 file(s)...
Processing validation set 'val_domain1' with 2 file(s)...
Processing validation set 'val_domain2' with 1 file(s)...

Epoch 1 | iter 1000 step 16 | loss train: 2.345, val: val_main: 2.456 | val_domain1: 2.789 | val_domain2: 3.012 | ...

iter 1000: val_main loss 2.4560
iter 1000: val_domain1 loss 2.7890
iter 1000: val_domain2 loss 3.0120
Total validation time: 850.45 ms
```

### WandB Metrics

When using multiple validation sets, metrics are logged with namespaces:

- `val_main/loss`, `val_main/ppl`
- `val_domain1/loss`, `val_domain1/ppl`
- `val_domain2/loss`, `val_domain2/ppl`
- `val_lang_en/loss`, `val_lang_en/ppl`
- `val_lang_es/loss`, `val_lang_es/ppl`

This allows you to track and compare performance across different validation sets in the same dashboard.

## Use Cases

### 1. Domain-Specific Evaluation

Track model performance on different text domains:

```yaml
val_data_path:
  val_news: /data/news_val
  val_code: /data/code_val
  val_medical: /data/medical_val
  val_legal: /data/legal_val
```

### 2. Multilingual Training

Monitor performance across languages:

```yaml
val_data_path:
  val_en: /data/english_val
  val_es: /data/spanish_val
  val_fr: /data/french_val
  val_de: /data/german_val
  val_zh: /data/chinese_val
```

### 3. Difficulty-Based Evaluation

Evaluate on different complexity levels:

```yaml
val_data_path:
  val_easy: /data/simple_texts
  val_medium: /data/moderate_texts
  val_hard: /data/complex_texts
```

### 4. Time-Based Evaluation

Track performance on data from different time periods:

```yaml
val_data_path:
  val_2020: /data/texts_2020
  val_2021: /data/texts_2021
  val_2022: /data/texts_2022
  val_2023: /data/texts_2023
```

## API Reference

### Class: TextFilesMultiVal

#### Parameters

- **train_data_path** (`Path`): Directory containing training `.txt` files
- **val_data_path** (`Optional[Union[Path, Dict[str, Path]]]`): 
  - `None`: Auto-split from training data
  - `Path`: Single validation directory
  - `Dict[str, Path]`: Multiple validation directories with custom names
- **seed** (`int`, default=42): Random seed for shuffling
- **num_workers** (`int`, default=4): Number of data loading workers

#### Methods

- **`train_dataloader()`**: Returns training DataLoader
- **`val_dataloader()`**: Returns single validation DataLoader (backward compatible)
- **`val_dataloaders()`**: Returns dict of all validation DataLoaders (preferred for multiple sets)

## Comparison with TextFiles

| Feature | TextFiles | TextFilesMultiVal |
|---------|-----------|-------------------|
| Single validation set | ✅ | ✅ |
| Multiple validation sets | ❌ | ✅ |
| Auto-split from training | ✅ | ✅ |
| Named validation sets | ❌ | ✅ |
| Backward compatible | - | ✅ |
| Custom metrics per set | ❌ | ✅ |

## Complete Example Config

```yaml
model_name: pythia-14m
out_dir: out/pretrain/comprehensive_eval
precision: bf16-mixed

data:
  class_path: litgpt.data.TextFilesMultiVal
  init_args:
    train_data_path: /data/train
    val_data_path:
      val_main: /data/val_main
      val_news: /data/val_news
      val_code: /data/val_code
      val_en: /data/val_english
      val_es: /data/val_spanish
    num_workers: 4

train:
  save_interval: 1000
  log_interval: 1
  global_batch_size: 512
  micro_batch_size: 4
  max_tokens: 1000000000
  max_norm: 1.0
  min_lr: 4e-5
  lr_warmup_steps: 2000

eval:
  interval: 1000
  max_iters: 100
  initial_validation: true
  final_validation: true

tokenizer_dir: checkpoints/EleutherAI/pythia-14m

logger_name: wandb
log:
  project: my-pretraining
  run: multi-domain-eval
```

## Implementation Details

### How It Works

1. **Initialization**: `__post_init__()` normalizes all validation paths to a dictionary format
2. **Data Preparation**: `prepare_data()` processes each validation set separately into optimized binary format
3. **DataLoader Creation**: 
   - `val_dataloader()`: Returns first validation set (backward compatible)
   - `val_dataloaders()`: Returns all validation sets as a dictionary
4. **Pretrain Integration**: The `pretrain.py` automatically detects and uses `val_dataloaders()` when available

### Preprocessed Data Location

Training data: `{train_data_path}/train/`  
Validation data: `{val_data_path[name]}/val/` for each validation set

Example:
```
/path/to/train/train/           # Preprocessed training data
/path/to/val_main/val/          # Preprocessed val_main data
/path/to/val_domain1/val/       # Preprocessed val_domain1 data
```

## Migration Guide

### From TextFiles to TextFilesMultiVal

**No changes needed!** Simply replace the class name:

```yaml
# Before
data:
  class_path: litgpt.data.TextFiles
  init_args:
    train_data_path: /path/to/train
    val_data_path: /path/to/val

# After (same behavior)
data:
  class_path: litgpt.data.TextFilesMultiVal
  init_args:
    train_data_path: /path/to/train
    val_data_path: /path/to/val
```

### Adding Multiple Validation Sets

Just change `val_data_path` to a dictionary:

```yaml
data:
  class_path: litgpt.data.TextFilesMultiVal
  init_args:
    train_data_path: /path/to/train
    val_data_path:
      val_main: /path/to/val
      val_new_domain: /path/to/new_domain
```

## Best Practices

1. **Name Validation Sets Clearly**: Use descriptive names like `val_news`, `val_code`, `val_en` instead of `val1`, `val2`
2. **Balance Validation Set Sizes**: Ensure validation sets have similar token counts for fair comparison
3. **Monitor All Metrics**: Track all validation sets in WandB to identify domain-specific issues
4. **Reasonable `max_iters`**: Set `eval.max_iters` appropriately based on validation set size
5. **Consistent Preprocessing**: Ensure all validation sets use the same tokenizer and preprocessing

## Troubleshooting

### Empty validation batches error

```
RuntimeError: stack expects a non-empty TensorList
```

**Solution**: Ensure `eval.max_iters` is set appropriately and validation directories contain `.txt` files.

### Validation set not found

```
AssertionError: No .txt files found in validation data 'val_name': /path
```

**Solution**: Verify the path contains `.txt` files and the path is correct.

### Memory issues with many validation sets

**Solution**: Reduce `eval.max_iters` or validate less frequently by increasing `eval.interval`.

## See Also

- [TextFiles Documentation](./text_files.py)
- [Pretrain Documentation](../pretrain.py)
- [Example Configs](../../tutorials/examples/multi_validation_data_module)
- [Multiple Validation Sets Guide](../../tutorials/examples/multi_validation_data_module/MULTI_VALIDATION_SETS.md)
