from typing import List, Union
from pathlib import Path
from torch.utils.data import Dataset

from vphoberttagger.processor import NerFeatures
from vphoberttagger.constant import PROCESSOR_MAPPING

import os
import torch

if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        torch.serialization.add_safe_globals([NerFeatures])
    except Exception:
        pass


def _safe_load_torch(f, **kwargs):
    try:
        return torch.load(f, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(f, **kwargs)


class NerDataset(Dataset):
    def __init__(self, features: List[NerFeatures], device: str = 'cpu'):
        self.examples = features
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: val for key, val in self.examples[index].__dict__.items()}


def build_dataset(data_dir: Union[str, os.PathLike],
                  tokenizer,
                  label2id: List[str],
                  header: List[str],
                  dtype: str = 'train',
                  max_seq_len: int = 256,
                  device: str = 'cpu',
                  use_crf: bool = True,
                  overwrite_data: bool = False) -> NerDataset:
    data_path = Path(data_dir)
    # Support both train_data.txt and train.txt naming conventions
    dfile_candidates = [
        data_path / f"{dtype}_data.txt",
        data_path / f"{dtype}.txt"
    ]
    dfile_path = None
    for candidate in dfile_candidates:
        if candidate.exists():
            dfile_path = candidate
            break
            
    if dfile_path is None:
        raise FileNotFoundError(f"Could not find {dtype} dataset file in {data_dir}. Looked for {[str(c) for c in dfile_candidates]}")

    process_key = tokenizer.name_or_path
    processor_fn = PROCESSOR_MAPPING.get(process_key, None)
    if processor_fn is None:
        if 'phobert' in str(process_key).lower():
            processor_fn = PROCESSOR_MAPPING.get('vinai/phobert-base', convert_word_segment_examples_features)
        else:
            raise KeyError(f"Processor not found for tokenizer '{process_key}'. Supported: {list(PROCESSOR_MAPPING.keys())}")

    model_tag = str(tokenizer.name_or_path).replace('/', '_').replace('\\', '_')
    cached_path = dfile_path.with_name(f"{dfile_path.stem}_{model_tag}.cached")
    legacy_cached_path = dfile_path.with_suffix('.cached')

    if not overwrite_data and cached_path.exists():
        features = _safe_load_torch(cached_path)
    elif not overwrite_data and legacy_cached_path.exists() and ('phobert-base' in model_tag):
        features = _safe_load_torch(legacy_cached_path)
    else:
        features = processor_fn(dfile_path, tokenizer, label2id, header, max_seq_len, use_crf=use_crf)
        torch.save(features, cached_path)
        
    return NerDataset(features=features, device=device)
