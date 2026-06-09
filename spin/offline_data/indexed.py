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

"""Tokenized offline preference dataset in the Megatron `.bin`/`.idx` + parquet-driver
layout produced by the swiss-ai `posttraining` project.

Disk layout under ``<root>``::

    <accepted_prefix>.bin / .idx   # chosen sequences, each = tokens[prompt + response]
    <rejected_prefix>.bin / .idx   # rejected sequences
    <parquet_name>                 # one row per pair: <chosen_index_col>, <rejected_index_col>
    manifest.json                  # optional: {"tokenizer_name_or_path": "..."}

The prompt/response boundary is recovered per pair as the token-level longest common
prefix of the paired chosen/rejected sequences (the prompt is shared, responses
diverge) — no extra metadata required.
"""

import json
import logging
import os

import numpy as np

from .base import OfflinePreferenceDataset, PreferenceExample, truncate_pref
from .indexed_dataset import IndexedDataset

logger = logging.getLogger(__name__)

_VALID_CONSISTENCY = {"off", "warn", "error"}

DEFAULT_ACCEPTED_PREFIX = "accepted"
DEFAULT_REJECTED_PREFIX = "rejected"
DEFAULT_PARQUET_NAME = "pairs.parquet"
DEFAULT_CHOSEN_INDEX_COL = "chosen_index"
DEFAULT_REJECTED_INDEX_COL = "rejected_index"


def _longest_common_prefix_len(a: np.ndarray, b: np.ndarray) -> int:
    """Return the length of the longest common (token id) prefix of two 1-D arrays."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    mismatch = a[:n] != b[:n]
    if not mismatch.any():
        return n  # one sequence is a prefix of the other (or they are identical)
    return int(np.argmax(mismatch))


class IndexedPreferenceDataset(OfflinePreferenceDataset):
    """Parquet-driven reader over two Megatron indexed datasets (accepted / rejected).

    ``__len__`` is the number of (kept) parquet rows; ``__getitem__`` returns a
    :class:`PreferenceExample` of token ids.
    """

    def __init__(
        self,
        root: str,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
        accepted_prefix: str = DEFAULT_ACCEPTED_PREFIX,
        rejected_prefix: str = DEFAULT_REJECTED_PREFIX,
        parquet_path: str | None = None,
        parquet_name: str = DEFAULT_PARQUET_NAME,
        chosen_index_col: str = DEFAULT_CHOSEN_INDEX_COL,
        rejected_index_col: str = DEFAULT_REJECTED_INDEX_COL,
        tokenizer=None,
        tokenizer_consistency: str = "warn",
        drop_empty: bool = True,
        max_samples: int = -1,
    ) -> None:
        if tokenizer_consistency not in _VALID_CONSISTENCY:
            raise ValueError(
                f"tokenizer_consistency must be one of {sorted(_VALID_CONSISTENCY)}, "
                f"got {tokenizer_consistency!r}."
            )
        self.root = root
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        self.accepted = IndexedDataset(os.path.join(root, accepted_prefix))
        self.rejected = IndexedDataset(os.path.join(root, rejected_prefix))

        parquet_path = parquet_path or os.path.join(root, parquet_name)
        # Driver is small (two integer columns); load eagerly with pandas (no datasets dep).
        import pandas as pd

        pairs = pd.read_parquet(parquet_path)
        for col in (chosen_index_col, rejected_index_col):
            if col not in pairs.columns:
                raise ValueError(
                    f"Parquet driver {parquet_path} is missing column '{col}'. "
                    f"Found columns: {list(pairs.columns)}"
                )
        self._chosen_idx = np.asarray(pairs[chosen_index_col], dtype=np.int64)
        self._rejected_idx = np.asarray(pairs[rejected_index_col], dtype=np.int64)

        self._validate()
        self._maybe_check_tokenizer(root, tokenizer, tokenizer_consistency)

        # Row indirection: drop pairs whose chosen/rejected response is empty (an empty
        # completion -> all-pad response -> meaningless / degenerate DPO term). NOTE: the
        # drop scan recomputes the LCP that __getitem__ recomputes — fine for a parquet
        # driver, but it does touch every sequence via mmap once at init.
        def _is_empty(row: int) -> bool:
            acc = self.accepted[int(self._chosen_idx[row])]
            rej = self.rejected[int(self._rejected_idx[row])]
            plen = _longest_common_prefix_len(acc, rej)
            return (len(acc) - plen <= 0) or (len(rej) - plen <= 0)

        self._rows = self.select_rows(
            len(self._chosen_idx), _is_empty, drop_empty, max_samples, "IndexedPreferenceDataset"
        )

        logger.info(
            "IndexedPreferenceDataset: %d pairs over accepted=%d / rejected=%d sequences.",
            len(self._rows),
            len(self.accepted),
            len(self.rejected),
        )

    def _validate(self) -> None:
        if len(self._chosen_idx) != len(self._rejected_idx):
            raise ValueError(
                "Parquet driver has mismatched index columns: "
                f"{len(self._chosen_idx)} chosen vs {len(self._rejected_idx)} rejected."
            )
        n_acc, n_rej = len(self.accepted), len(self.rejected)
        if len(self._chosen_idx) and (self._chosen_idx.min() < 0 or self._chosen_idx.max() >= n_acc):
            raise ValueError(
                f"chosen_index out of bounds for accepted dataset of length {n_acc} "
                f"(range [{self._chosen_idx.min()}, {self._chosen_idx.max()}])."
            )
        if len(self._rejected_idx) and (self._rejected_idx.min() < 0 or self._rejected_idx.max() >= n_rej):
            raise ValueError(
                f"rejected_index out of bounds for rejected dataset of length {n_rej} "
                f"(range [{self._rejected_idx.min()}, {self._rejected_idx.max()}])."
            )

    def _maybe_check_tokenizer(self, root, tokenizer, tokenizer_consistency: str) -> None:
        """Compare the producer's tokenizer (manifest.json, if present) to the training one."""
        if tokenizer_consistency == "off" or tokenizer is None:
            return
        manifest_path = os.path.join(root, "manifest.json")
        if not os.path.exists(manifest_path):
            return
        with open(manifest_path) as f:
            manifest = json.load(f)
        produced = manifest.get("tokenizer_name_or_path")
        current = getattr(tokenizer, "name_or_path", None)
        if produced is not None and current is not None and produced != current:
            msg = (
                f"Tokenized data was produced with tokenizer '{produced}' but training uses "
                f"'{current}'. Token ids may be incompatible."
            )
            if tokenizer_consistency == "error":
                raise ValueError(msg)
            logger.warning(msg)

    @classmethod
    def from_config(cls, data_cfg, path: str, tokenizer) -> "IndexedPreferenceDataset":
        return cls(
            root=path,
            max_prompt_length=data_cfg.get("max_prompt_length", None),
            max_response_length=data_cfg.get("max_response_length", None),
            accepted_prefix=data_cfg.get("offpolicy_accepted_prefix", DEFAULT_ACCEPTED_PREFIX),
            rejected_prefix=data_cfg.get("offpolicy_rejected_prefix", DEFAULT_REJECTED_PREFIX),
            parquet_name=data_cfg.get("offpolicy_parquet_name", DEFAULT_PARQUET_NAME),
            chosen_index_col=data_cfg.get("offpolicy_chosen_index_col", DEFAULT_CHOSEN_INDEX_COL),
            rejected_index_col=data_cfg.get("offpolicy_rejected_index_col", DEFAULT_REJECTED_INDEX_COL),
            tokenizer=tokenizer,
            tokenizer_consistency=data_cfg.get("offpolicy_tokenizer_consistency", "warn"),
            max_samples=data_cfg.get("offpolicy_max_samples", -1),
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> PreferenceExample:
        row = int(self._rows[idx])
        accepted_tokens = self.accepted[int(self._chosen_idx[row])]
        rejected_tokens = self.rejected[int(self._rejected_idx[row])]

        plen = _longest_common_prefix_len(accepted_tokens, rejected_tokens)
        prompt_ids, chosen_ids, rejected_ids = truncate_pref(
            accepted_tokens[:plen],
            accepted_tokens[plen:],
            rejected_tokens[plen:],
            self.max_prompt_length,
            self.max_response_length,
        )
        return PreferenceExample(
            prompt_input_ids=prompt_ids,
            chosen_input_ids=chosen_ids,
            rejected_input_ids=rejected_ids,
        )
