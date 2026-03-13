# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
import time
from pathlib import Path

from litgpt.tokenizer import Tokenizer
from litgpt.utils import CLI, extend_checkpoint_dir
from litgpt.utils import _LITDATA_AVAILABLE, CLI, extend_checkpoint_dir

if _LITDATA_AVAILABLE:
    from litdata.processing.data_processor import DataChunkRecipe
else:
    DataChunkRecipe = object

class PPDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(self, tokenizer: Tokenizer, chunk_size: int, text_field: str = "text"):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer
        self.text_field = text_field

    def prepare_structure(self, input_dir):
        files = list(Path(input_dir).rglob("*.zst")) + list(Path(input_dir).rglob("*.jsonl"))
        return [str(file) for file in files]

    def prepare_item(self, filepath):
        import zstandard as zstd

        filepath = Path(filepath)
        
        # Handle compressed or uncompressed files
        if filepath.suffix == ".zst":
            file_handle = zstd.open(open(filepath, "rb"), "rt", encoding="utf-8")
        else:  # .jsonl
            file_handle = open(filepath, "r", encoding="utf-8")
        
        with file_handle as f:
            for row in f:
                text = json.loads(row)[self.text_field]
                text_ids = self.tokenizer.encode(string=text, bos=False, eos=True)
                yield text_ids


def prepare(
    input_dir: Path = Path("indata/train"),
    output_dir: Path = Path("outdata/train"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf/"),
    chunk_size: int = (4097 * 16384),
    fast_dev_run: bool = False,
    text_field: str = "text",
) -> None:
    from litdata.processing.data_processor import DataProcessor
    from litdata.streaming.item_loader import TokensLoader

    tokenizer_path = extend_checkpoint_dir(tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = PPDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size, text_field=text_field)
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count(),
        num_downloaders=1,
        item_loader=TokensLoader(),
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    CLI(prepare)
