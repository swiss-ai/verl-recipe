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

"""Offline preference pairs stored as TEXT chat messages (the `text_chat` format),
read into token-id `PreferenceExample`s so the text format can participate in the
`offpolicy_datasets` mixture: per-dataset `max_samples`/`selection`, and mixing with the
pre-tokenized `indexed`/`single_parquet` formats.

Each parquet row holds:
    <prompt_key>            : list[ {role, content} ]  (chat messages)
    <chosen_response_key>   : { role, content }        (single assistant message)
    <rejected_response_key> : { role, content }

The per-row prompt/response tokenization mirrors the trainer's `tokenize_offpolicy_pairs`
exactly (prompt = `apply_chat_template(prompt, add_generation_prompt=True)`, response =
the suffix of `apply_chat_template(prompt + [response])` after the prompt tokens), so a
`text_chat` entry assembles to the same layout as the pre-tokenized formats.

NOTE: the trainer's *single-source* legacy text path (`offpolicy_files` +
`offpolicy_format=text_chat`, no mixture) still uses verl's `RLHFDataset`; this class is
used for `text_chat` entries inside `offpolicy_datasets`.
"""

import logging

import numpy as np

from .base import OfflinePreferenceDataset, PreferenceExample, truncate_pref

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_KEY = "prompt"
DEFAULT_CHOSEN_RESPONSE_KEY = "chosen_response"
DEFAULT_REJECTED_RESPONSE_KEY = "rejected_response"


def _message_content(msg) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content", "") or "")
    return str(msg or "")


class TextChatPreferenceDataset(OfflinePreferenceDataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
        prompt_key: str = DEFAULT_PROMPT_KEY,
        chosen_response_key: str = DEFAULT_CHOSEN_RESPONSE_KEY,
        rejected_response_key: str = DEFAULT_REJECTED_RESPONSE_KEY,
        drop_empty: bool = True,
        max_samples: int = -1,
        selection: str = "head",
        seed: int | None = None,
    ) -> None:
        import pandas as pd

        if tokenizer is None:
            raise ValueError("TextChatPreferenceDataset requires a tokenizer (to tokenize the chat text).")
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        df = pd.read_parquet(parquet_path)
        for col in (prompt_key, chosen_response_key, rejected_response_key):
            if col not in df.columns:
                raise ValueError(
                    f"Parquet {parquet_path} is missing column '{col}'. Found columns: {list(df.columns)}"
                )
        self._prompt = df[prompt_key].tolist()
        self._chosen = df[chosen_response_key].tolist()
        self._rejected = df[rejected_response_key].tolist()

        # Empty check is on the raw text content (cheap — no tokenization needed).
        self._rows = self.select_rows(
            len(self._prompt),
            lambda i: not _message_content(self._chosen[i]).strip()
            or not _message_content(self._rejected[i]).strip(),
            drop_empty,
            max_samples,
            "TextChatPreferenceDataset",
            selection=selection,
            seed=seed,
        )

    @classmethod
    def from_config(
        cls, data_cfg, path: str, tokenizer, *, max_samples=None, selection=None, seed=None
    ) -> "TextChatPreferenceDataset":
        return cls(
            parquet_path=path,
            tokenizer=tokenizer,
            max_prompt_length=data_cfg.get("max_prompt_length", None),
            max_response_length=data_cfg.get("max_response_length", None),
            prompt_key=data_cfg.get("offpolicy_prompt_key", DEFAULT_PROMPT_KEY),
            chosen_response_key=data_cfg.get("offpolicy_chosen_response_key", DEFAULT_CHOSEN_RESPONSE_KEY),
            rejected_response_key=data_cfg.get("offpolicy_rejected_response_key", DEFAULT_REJECTED_RESPONSE_KEY),
            max_samples=max_samples if max_samples is not None else data_cfg.get("offpolicy_max_samples", -1),
            selection=selection if selection is not None else data_cfg.get("offpolicy_selection", "head"),
            seed=seed,
        )

    @staticmethod
    def _as_messages(prompt_field) -> list:
        # Parquet list-columns may come back as a numpy array of message dicts or a
        # python list; normalize to a plain list of dicts either way.
        if isinstance(prompt_field, np.ndarray):
            return prompt_field.tolist()
        return list(prompt_field)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> PreferenceExample:
        row = int(self._rows[idx])
        raw_prompt = self._as_messages(self._prompt[row])
        tok = self.tokenizer

        prompt_text = tok.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

        def _response_ids(msg) -> list[int]:
            full_text = tok.apply_chat_template(
                list(raw_prompt) + [msg], add_generation_prompt=False, tokenize=False
            )
            full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
            return full_ids[len(prompt_ids):]

        chosen_ids = _response_ids(self._chosen[row])
        rejected_ids = _response_ids(self._rejected[row])

        prompt_ids, chosen_ids, rejected_ids = truncate_pref(
            prompt_ids, chosen_ids, rejected_ids, self.max_prompt_length, self.max_response_length
        )
        return PreferenceExample(
            prompt_input_ids=prompt_ids,
            chosen_input_ids=chosen_ids,
            rejected_input_ids=rejected_ids,
        )
