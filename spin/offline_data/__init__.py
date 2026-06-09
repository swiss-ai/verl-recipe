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

import logging
from collections import defaultdict

import numpy as np
import torch

from .base import (
    CHOSEN_RESPONSE_IDS_KEY,
    REJECTED_RESPONSE_IDS_KEY,
    ConcatPreferenceDataset,
    OfflinePreferenceDataset,
    PreferenceExample,
)
from .indexed import IndexedPreferenceDataset
from .single_parquet import SingleParquetPreferenceDataset
from .text_chat import TextChatPreferenceDataset

logger = logging.getLogger(__name__)

# NOTE: verl (DataProto, postprocess_data) is imported lazily — see the collate fn and
# the module-level __getattr__ for assemble_offpolicy_pairs — so that constructing
# offline datasets (and unit-testing the data layer) does not require verl.

__all__ = [
    "OfflinePreferenceDataset",
    "PreferenceExample",
    "IndexedPreferenceDataset",
    "SingleParquetPreferenceDataset",
    "TextChatPreferenceDataset",
    "ConcatPreferenceDataset",
    "build_offline_dataset",
    "make_tokenized_offpolicy_collate_fn",
    "assemble_offpolicy_pairs",
    "OFFLINE_FORMATS",
]

# format name -> dataset class. NOTE: a SINGLE-source text_chat dataset
# (offpolicy_files + offpolicy_format=text_chat, no mixture) is still served by the
# legacy RLHFDataset path in spin_trainer._create_dataloader for backward compatibility.
# The TextChatPreferenceDataset entry here is used for text_chat datasets listed under
# `offpolicy_datasets` (per-dataset max_samples/selection, and mixing with tokenized
# formats).
OFFLINE_FORMATS = {
    "indexed": IndexedPreferenceDataset,
    "single_parquet": SingleParquetPreferenceDataset,
    "text_chat": TextChatPreferenceDataset,
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


def _resolve_format(fmt):
    if fmt not in OFFLINE_FORMATS:
        raise ValueError(f"Unknown offline format={fmt!r}. Supported formats: {sorted(OFFLINE_FORMATS)}.")
    return OFFLINE_FORMATS[fmt]


def build_offline_dataset(data_cfg, tokenizer) -> OfflinePreferenceDataset:
    """Construct the offline preference dataset(s) for the off-policy DPO stream.

    Two modes:
    - **single** (``data.offpolicy_files``): one dataset of ``data.offpolicy_format``.
    - **mixture** (``data.offpolicy_datasets``): a list of entries, each
      ``{path, format?, max_samples?, selection?}``, built independently (per-dataset
      ``max_samples``/``selection`` applied *before* combining) and pooled via
      :class:`ConcatPreferenceDataset`. Per-format column/prefix knobs are read from the
      global ``data`` config and shared across entries of that format.

    Dispatches purely through ``OFFLINE_FORMATS`` + each class's ``from_config`` — so a
    new format needs only a subclass (implementing ``from_config``) and a registry entry.
    """
    default_fmt = data_cfg.get("offpolicy_format", "text_chat")
    entries = data_cfg.get("offpolicy_datasets", None)

    if entries and data_cfg.get("offpolicy_files", None):
        logger.warning(
            "Both data.offpolicy_datasets and data.offpolicy_files are set; using the "
            "offpolicy_datasets mixture and ignoring offpolicy_files."
        )

    if entries:
        # Deterministic base seed so 'random' subsets are stable across runs/resumes.
        base_seed = data_cfg.get("seed", None)
        base_seed = 42 if base_seed is None else int(base_seed)
        global_selection = data_cfg.get("offpolicy_selection", "head")
        subsets = []
        for i, entry in enumerate(entries):
            path = entry.get("path", None)
            if not path:
                raise ValueError(f"data.offpolicy_datasets[{i}] is missing required key 'path'.")
            cls = _resolve_format(entry.get("format", default_fmt))
            subsets.append(
                cls.from_config(
                    data_cfg,
                    path,
                    tokenizer,
                    max_samples=entry.get("max_samples", -1),
                    selection=entry.get("selection", global_selection),
                    seed=base_seed + i,
                )
            )
        return ConcatPreferenceDataset(subsets)

    # single-dataset mode
    cls = _resolve_format(default_fmt)
    path = _as_single_path(data_cfg.get("offpolicy_files", None))
    if not path:
        raise ValueError("data.offpolicy_files (or data.offpolicy_datasets) must be set to build an offline dataset.")
    return cls.from_config(data_cfg, path, tokenizer)


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
