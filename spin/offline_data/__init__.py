# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pluggable PRE-TOKENIZED offline preference datasets for spin's off-policy DPO path.

Public API:
    - ``build_offline_dataset(data_cfg, tokenizer)`` — factory keyed by
      ``data.offpolicy_format``.
    - ``make_tokenized_offpolicy_collate_fn(...)`` — collate that turns
      ``PreferenceExample``s into a dict consumable by ``DataProto.from_single_dict``.
    - ``assemble_offpolicy_pairs(...)`` — tokenizer-free twin of
      ``tokenize_offpolicy_pairs`` (lazily exposed via module ``__getattr__`` so that
      importing this package does not require verl).

To add a format: implement an ``OfflinePreferenceDataset`` subclass and register it in
``OFFLINE_FORMATS`` below. No trainer changes needed.
"""

from collections import defaultdict

import numpy as np
import torch

from .base import (
    CHOSEN_RESPONSE_IDS_KEY,
    REJECTED_RESPONSE_IDS_KEY,
    OfflinePreferenceDataset,
    PreferenceExample,
)
from .indexed import IndexedPreferenceDataset
from .single_parquet import SingleParquetPreferenceDataset

# NOTE: verl (DataProto, postprocess_data) is imported lazily — see the collate fn and
# the module-level __getattr__ for assemble_offpolicy_pairs — so that constructing
# offline datasets (and unit-testing the data layer) does not require verl.

__all__ = [
    "OfflinePreferenceDataset",
    "PreferenceExample",
    "IndexedPreferenceDataset",
    "SingleParquetPreferenceDataset",
    "build_offline_dataset",
    "make_tokenized_offpolicy_collate_fn",
    "assemble_offpolicy_pairs",
    "OFFLINE_FORMATS",
]

# format name -> dataset class. "text_chat" is NOT here: it is the legacy RLHFDataset
# path handled directly in spin_trainer._create_dataloader.
OFFLINE_FORMATS = {
    "indexed": IndexedPreferenceDataset,
    "single_parquet": SingleParquetPreferenceDataset,
}


def _as_single_path(offpolicy_files):
    """``offpolicy_files`` may be a str or a 1-element list; tokenized formats use a
    single root dir / parquet path."""
    if isinstance(offpolicy_files, (list, tuple)):
        if len(offpolicy_files) != 1:
            raise ValueError(
                "Tokenized offline formats expect a single path in data.offpolicy_files, "
                f"got {len(offpolicy_files)}: {list(offpolicy_files)}"
            )
        return offpolicy_files[0]
    return offpolicy_files


def build_offline_dataset(data_cfg, tokenizer) -> OfflinePreferenceDataset:
    """Construct the offline preference dataset selected by ``data_cfg.offpolicy_format``.

    Dispatches purely through ``OFFLINE_FORMATS`` + each class's ``from_config`` — so a
    new format needs only a subclass (implementing ``from_config``) and a registry entry.
    """
    fmt = data_cfg.get("offpolicy_format", "text_chat")
    if fmt not in OFFLINE_FORMATS:
        raise ValueError(
            f"Unknown data.offpolicy_format={fmt!r}. "
            f"Supported tokenized formats: {sorted(OFFLINE_FORMATS)} (or 'text_chat')."
        )
    path = _as_single_path(data_cfg.get("offpolicy_files", None))
    if not path:
        raise ValueError("data.offpolicy_files must be set to build an offline dataset.")
    return OFFLINE_FORMATS[fmt].from_config(data_cfg, path, tokenizer)


def make_tokenized_offpolicy_collate_fn(tokenizer, max_prompt_length: int, truncation: str = "error"):
    """Collate ``PreferenceExample``s into a dict for ``DataProto.from_single_dict``.

    Produces:
        tensors:     input_ids / attention_mask  [N, max_prompt_length] (prompt, left-padded)
        non-tensors: chosen_response_ids / rejected_response_ids  (object arrays of int lists)

    ``assemble_offpolicy_pairs`` consumes exactly these to build the 2N-row DataProto.
    """
    from verl.utils.torch_functional import postprocess_data

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _collate(examples: list[PreferenceExample]) -> dict:
        tensors = defaultdict(list)
        chosen_ids_list = []
        rejected_ids_list = []

        for ex in examples:
            p = torch.tensor(ex.prompt_input_ids, dtype=torch.long).unsqueeze(0)  # [1, L]
            p_mask = torch.ones_like(p)
            p, p_mask = postprocess_data(
                input_ids=p,
                attention_mask=p_mask,
                max_length=max_prompt_length,
                pad_token_id=pad_id,
                left_pad=True,
                truncation=truncation,
            )
            tensors["input_ids"].append(p[0])
            tensors["attention_mask"].append(p_mask[0])
            chosen_ids_list.append(list(ex.chosen_input_ids))
            rejected_ids_list.append(list(ex.rejected_input_ids))

        out = {
            "input_ids": torch.stack(tensors["input_ids"], dim=0),
            "attention_mask": torch.stack(tensors["attention_mask"], dim=0),
        }
        # object arrays so DataProto keeps them as ragged non-tensors
        chosen_arr = np.empty(len(chosen_ids_list), dtype=object)
        rejected_arr = np.empty(len(rejected_ids_list), dtype=object)
        for i in range(len(chosen_ids_list)):
            chosen_arr[i] = chosen_ids_list[i]
            rejected_arr[i] = rejected_ids_list[i]
        out[CHOSEN_RESPONSE_IDS_KEY] = chosen_arr
        out[REJECTED_RESPONSE_IDS_KEY] = rejected_arr
        return out

    return _collate


def __getattr__(name):
    # Lazily expose the verl-dependent assembler so that `import ...offline_data` and
    # dataset construction work without verl installed.
    if name == "assemble_offpolicy_pairs":
        from .assemble import assemble_offpolicy_pairs

        return assemble_offpolicy_pairs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
