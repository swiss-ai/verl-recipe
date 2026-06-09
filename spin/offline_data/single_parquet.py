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

"""Tokenized offline preference dataset stored as a SINGLE parquet file whose columns
already hold token-id lists — one row per preference pair.

Default columns (overridable):
    prompt_input_ids    : list[int]  shared prompt token ids
    chosen_input_ids    : list[int]  chosen RESPONSE token ids (post-prompt)
    rejected_input_ids  : list[int]  rejected RESPONSE token ids (post-prompt)

This is the "all-in-one parquet" alternative to the Megatron `.bin/.idx` layout; it
exists to demonstrate that new on-disk formats plug in by subclassing
``OfflinePreferenceDataset`` with no trainer changes.
"""

import logging

from .base import OfflinePreferenceDataset, PreferenceExample, truncate_pref

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_COL = "prompt_input_ids"
DEFAULT_CHOSEN_COL = "chosen_input_ids"
DEFAULT_REJECTED_COL = "rejected_input_ids"


class SingleParquetPreferenceDataset(OfflinePreferenceDataset):
    def __init__(
        self,
        parquet_path: str,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
        prompt_col: str = DEFAULT_PROMPT_COL,
        chosen_col: str = DEFAULT_CHOSEN_COL,
        rejected_col: str = DEFAULT_REJECTED_COL,
        drop_empty: bool = True,
        max_samples: int = -1,
        selection: str = "head",
        seed: int | None = None,
    ) -> None:
        import pandas as pd

        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.prompt_col = prompt_col
        self.chosen_col = chosen_col
        self.rejected_col = rejected_col

        df = pd.read_parquet(parquet_path)
        for col in (prompt_col, chosen_col, rejected_col):
            if col not in df.columns:
                raise ValueError(
                    f"Parquet {parquet_path} is missing column '{col}'. "
                    f"Found columns: {list(df.columns)}"
                )
        self._prompt = df[prompt_col].tolist()
        self._chosen = df[chosen_col].tolist()
        self._rejected = df[rejected_col].tolist()

        self._rows = self.select_rows(
            len(self._prompt),
            lambda i: len(self._chosen[i]) == 0 or len(self._rejected[i]) == 0,
            drop_empty,
            max_samples,
            "SingleParquetPreferenceDataset",
            selection=selection,
            seed=seed,
        )

    @classmethod
    def from_config(
        cls, data_cfg, path: str, tokenizer, *, max_samples=None, selection=None, seed=None
    ) -> "SingleParquetPreferenceDataset":
        return cls(
            parquet_path=path,
            max_prompt_length=data_cfg.get("max_prompt_length", None),
            max_response_length=data_cfg.get("max_response_length", None),
            prompt_col=data_cfg.get("offpolicy_prompt_col", DEFAULT_PROMPT_COL),
            chosen_col=data_cfg.get("offpolicy_chosen_col", DEFAULT_CHOSEN_COL),
            rejected_col=data_cfg.get("offpolicy_rejected_col", DEFAULT_REJECTED_COL),
            max_samples=max_samples if max_samples is not None else data_cfg.get("offpolicy_max_samples", -1),
            selection=selection if selection is not None else data_cfg.get("offpolicy_selection", "head"),
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> PreferenceExample:
        row = int(self._rows[idx])
        # truncate_pref handles both python lists and numpy-array parquet cells.
        prompt_ids, chosen_ids, rejected_ids = truncate_pref(
            self._prompt[row],
            self._chosen[row],
            self._rejected[row],
            self.max_prompt_length,
            self.max_response_length,
        )
        return PreferenceExample(
            prompt_input_ids=prompt_ids,
            chosen_input_ids=chosen_ids,
            rejected_input_ids=rejected_ids,
        )
