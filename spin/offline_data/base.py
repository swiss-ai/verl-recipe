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

"""Base abstractions for loading PRE-TOKENIZED offline preference data into spin's
off-policy DPO path.

A concrete dataset reads some on-disk layout (Megatron .bin/.idx, a single parquet,
...) and yields a uniform :class:`PreferenceExample` of *token ids* — already split
into a shared prompt and the chosen/rejected responses. The recipe's collate +
``assemble_offpolicy_pairs`` then turn a batch of these into the exact DataProto the
trainer's off-policy block expects, with NO re-tokenization.

To add a new on-disk format: subclass :class:`OfflinePreferenceDataset`, return
``PreferenceExample``s, and register it in ``offline_data/__init__.py``. No trainer
changes are required.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

# Non-tensor batch keys carrying per-example response token ids between the collate fn
# and ``assemble_offpolicy_pairs``. Defined here (verl-free) so both sides can import
# them without pulling in the verl-dependent assembly module.
CHOSEN_RESPONSE_IDS_KEY = "chosen_response_ids"
REJECTED_RESPONSE_IDS_KEY = "rejected_response_ids"


@dataclass
class PreferenceExample:
    """One offline preference pair as token ids.

    Attributes:
        prompt_input_ids: token ids of the shared prompt (no padding).
        chosen_input_ids: token ids of the chosen RESPONSE only (post-prompt).
        rejected_input_ids: token ids of the rejected RESPONSE only (post-prompt).
        ref_chosen_logps / ref_rejected_logps: optional precomputed reference
            sequence log-probs. NOTE: spin recomputes reference log-probs with its
            dynamic reference model every step, so these are currently ignored by the
            trainer; they are carried only for forward compatibility.
    """

    prompt_input_ids: list[int]
    chosen_input_ids: list[int]
    rejected_input_ids: list[int]
    ref_chosen_logps: Optional[float] = field(default=None)
    ref_rejected_logps: Optional[float] = field(default=None)


def truncate_pref(prompt, chosen, rejected, max_prompt_length, max_response_length):
    """Left-truncate the prompt and right-truncate the responses, returning int lists.

    Accepts python lists or numpy arrays (both slice the same way). Shared by every
    format so the truncation rule stays identical across subclasses (important for
    cross-format parity).
    """
    if max_prompt_length is not None:
        prompt = prompt[-max_prompt_length:]
    if max_response_length is not None:
        chosen = chosen[:max_response_length]
        rejected = rejected[:max_response_length]
    to_list = lambda x: x.tolist() if isinstance(x, np.ndarray) else [int(t) for t in x]
    return to_list(prompt), to_list(chosen), to_list(rejected)


class OfflinePreferenceDataset(Dataset, ABC):
    """Map-style dataset yielding :class:`PreferenceExample` of token ids.

    Subclasses own the on-disk reading and implement :meth:`from_config` so the
    factory in ``offline_data/__init__.py`` can build them purely from the registry,
    keyed by ``data.offpolicy_format`` — adding a format requires only a new subclass
    plus a registry entry, with no trainer changes.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, data_cfg, path: str, tokenizer) -> "OfflinePreferenceDataset":
        """Construct the dataset from the recipe ``data`` config and the offline path.

        ``path`` is ``data.offpolicy_files`` resolved to a single str (a root dir or a
        parquet path, depending on the format).
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> PreferenceExample: ...

    @staticmethod
    def select_rows(
        n: int, is_empty: Optional[Callable[[int], bool]], drop_empty: bool, max_samples: int, name: str
    ) -> np.ndarray:
        """Build the kept-row index array: optionally drop empty pairs, then cap to
        ``max_samples``. Shared by all formats. ``is_empty(i)`` reports whether row ``i``
        has an empty chosen/rejected response.
        """
        import logging

        if drop_empty and is_empty is not None:
            kept = [i for i in range(n) if not is_empty(i)]
            n_dropped = n - len(kept)
            if n_dropped:
                logging.getLogger(__name__).warning(
                    "%s: dropped %d/%d pairs with an empty chosen/rejected response.",
                    name,
                    n_dropped,
                    n,
                )
        else:
            kept = list(range(n))
        rows = np.asarray(kept, dtype=np.int64)
        if max_samples is not None and max_samples >= 0:
            rows = rows[:max_samples]
        return rows
