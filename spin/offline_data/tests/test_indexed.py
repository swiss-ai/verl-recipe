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

"""Tests for the pre-tokenized offline preference data layer.

The data-layer tests (IndexedPreferenceDataset / SingleParquetPreferenceDataset /
factory) require only numpy + pandas + torch. The collate + assembly tests additionally
require verl (DataProto / postprocess_data) and are skipped when verl is unavailable.

Run:  pytest spin/offline_data/tests/test_indexed.py
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable so `spin.offline_data` resolves when running standalone.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spin.offline_data import (  # noqa: E402
    IndexedPreferenceDataset,
    SingleParquetPreferenceDataset,
    build_offline_dataset,
)
from spin.offline_data.base import PreferenceExample  # noqa: E402

# --- Megatron .bin/.idx writer (ported from posttraining's test helper) ---
_INDEX_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_CODE = {np.uint16: 8, np.int32: 4, np.int64: 5}


def _write_megatron_indexed(prefix, sequences, dtype=np.int32):
    """Write a minimal Megatron indexed dataset (one sequence == one document)."""
    seq_lengths = [len(s) for s in sequences]
    itemsize = np.dtype(dtype).itemsize
    pointers, curr = [], 0
    for length in seq_lengths:
        pointers.append(curr)
        curr += length * itemsize
    document_indices = list(range(len(sequences) + 1))

    with open(prefix + ".bin", "wb") as f:
        for s in sequences:
            f.write(np.array(s, dtype=dtype).tobytes(order="C"))

    with open(prefix + ".idx", "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", _DTYPE_CODE[dtype]))
        f.write(struct.pack("<Q", len(sequences)))
        f.write(struct.pack("<Q", len(document_indices)))
        f.write(np.array(seq_lengths, dtype=np.int32).tobytes(order="C"))
        f.write(np.array(pointers, dtype=np.int64).tobytes(order="C"))
        f.write(np.array(document_indices, dtype=np.int64).tobytes(order="C"))


def _build_indexed_fixture(root: Path, chosen_index=(1, 0), rejected_index=(1, 0), manifest=None):
    # accepted/rejected sequences = tokens[prompt + response]
    accepted = [
        [10, 11, 12, 20, 21],              # prompt [10,11,12] + chosen [20,21]
        [40, 41, 50, 51, 52, 53, 54, 55],  # prompt [40,41]    + chosen [50,51,52,53,54,55]
    ]
    rejected = [
        [10, 11, 12, 30, 31, 32],          # prompt [10,11,12] + rejected [30,31,32]
        [40, 41, 60, 61],                  # prompt [40,41]    + rejected [60,61]
    ]
    _write_megatron_indexed(str(root / "accepted"), accepted)
    _write_megatron_indexed(str(root / "rejected"), rejected)
    pd.DataFrame(
        {"chosen_index": list(chosen_index), "rejected_index": list(rejected_index)}
    ).to_parquet(str(root / "pairs.parquet"))
    if manifest is not None:
        (root / "manifest.json").write_text(json.dumps(manifest))


class _FakeTokenizer:
    pad_token_id = 999
    eos_token_id = 999
    name_or_path = "fake/tokenizer"


# ----------------------------- data layer (no verl) -----------------------------


def test_indexed_lcp_split_and_pairing(tmp_path):
    _build_indexed_fixture(tmp_path)  # chosen_index/rejected_index = [1, 0]
    ds = IndexedPreferenceDataset(root=str(tmp_path), max_response_length=4)
    assert len(ds) == 2

    # Row 0 -> accepted[1]/rejected[1]; shared prefix [40,41]; chosen truncated to 4.
    ex0 = ds[0]
    assert isinstance(ex0, PreferenceExample)
    assert ex0.prompt_input_ids == [40, 41]
    assert ex0.chosen_input_ids == [50, 51, 52, 53]
    assert ex0.rejected_input_ids == [60, 61]

    # Row 1 -> accepted[0]/rejected[0]; shared prefix [10,11,12].
    ex1 = ds[1]
    assert ex1.prompt_input_ids == [10, 11, 12]
    assert ex1.chosen_input_ids == [20, 21]
    assert ex1.rejected_input_ids == [30, 31, 32]


def test_indexed_max_prompt_length_left_truncates(tmp_path):
    _build_indexed_fixture(tmp_path)
    ds = IndexedPreferenceDataset(root=str(tmp_path), max_prompt_length=2)
    # Row 1 prompt [10,11,12] left-truncated to last 2 -> [11,12].
    assert ds[1].prompt_input_ids == [11, 12]


def test_indexed_drop_empty_pair(tmp_path):
    # Make accepted[k]==rejected[k] for one pair so the response is empty (LCP == full).
    accepted = [[10, 11, 12, 20, 21], [40, 41, 42]]
    rejected = [[10, 11, 12, 30, 31, 32], [40, 41, 42]]  # 2nd pair identical -> empty resp
    _write_megatron_indexed(str(tmp_path / "accepted"), accepted)
    _write_megatron_indexed(str(tmp_path / "rejected"), rejected)
    pd.DataFrame({"chosen_index": [0, 1], "rejected_index": [0, 1]}).to_parquet(
        str(tmp_path / "pairs.parquet")
    )
    ds = IndexedPreferenceDataset(root=str(tmp_path))
    assert len(ds) == 1  # the empty pair (row 1) is dropped
    assert ds[0].prompt_input_ids == [10, 11, 12]


def test_indexed_out_of_bounds_index_raises(tmp_path):
    _build_indexed_fixture(tmp_path, chosen_index=(2, 0), rejected_index=(0, 0))  # 2 is OOB
    with pytest.raises(ValueError, match="out of bounds"):
        IndexedPreferenceDataset(root=str(tmp_path))


def test_indexed_tokenizer_mismatch_errors(tmp_path):
    _build_indexed_fixture(tmp_path, manifest={"tokenizer_name_or_path": "other/tok"})
    with pytest.raises(ValueError, match="incompatible"):
        IndexedPreferenceDataset(
            root=str(tmp_path), tokenizer=_FakeTokenizer(), tokenizer_consistency="error"
        )


def test_indexed_tokenizer_match_ok(tmp_path):
    _build_indexed_fixture(tmp_path, manifest={"tokenizer_name_or_path": "fake/tokenizer"})
    ds = IndexedPreferenceDataset(
        root=str(tmp_path), tokenizer=_FakeTokenizer(), tokenizer_consistency="error"
    )
    assert len(ds) == 2


def test_single_parquet_roundtrip(tmp_path):
    path = str(tmp_path / "pairs.parquet")
    pd.DataFrame(
        {
            "prompt_input_ids": [[1, 2, 3], [4, 5]],
            "chosen_input_ids": [[6, 7, 8, 9], [10]],
            "rejected_input_ids": [[11], [12, 13]],
        }
    ).to_parquet(path)
    ds = SingleParquetPreferenceDataset(parquet_path=path, max_response_length=2)
    assert len(ds) == 2
    ex0 = ds[0]
    assert ex0.prompt_input_ids == [1, 2, 3]
    assert ex0.chosen_input_ids == [6, 7]  # truncated to 2
    assert ex0.rejected_input_ids == [11]


def test_indexed_uint16_dtype(tmp_path):
    # Real swiss-ai data is typically uint16; ensure the dtype flows through to plain ints.
    accepted = [[10, 11, 12, 20, 21]]
    rejected = [[10, 11, 12, 30, 31]]
    _write_megatron_indexed(str(tmp_path / "accepted"), accepted, dtype=np.uint16)
    _write_megatron_indexed(str(tmp_path / "rejected"), rejected, dtype=np.uint16)
    pd.DataFrame({"chosen_index": [0], "rejected_index": [0]}).to_parquet(str(tmp_path / "pairs.parquet"))
    ds = IndexedPreferenceDataset(root=str(tmp_path))
    ex = ds[0]
    assert ex.prompt_input_ids == [10, 11, 12]
    assert ex.chosen_input_ids == [20, 21]
    assert ex.rejected_input_ids == [30, 31]
    assert all(isinstance(t, int) for t in ex.prompt_input_ids + ex.chosen_input_ids)


def test_indexed_max_samples(tmp_path):
    _build_indexed_fixture(tmp_path)
    ds = IndexedPreferenceDataset(root=str(tmp_path), max_samples=1)
    assert len(ds) == 1


def test_indexed_invalid_consistency_raises(tmp_path):
    _build_indexed_fixture(tmp_path)
    with pytest.raises(ValueError, match="tokenizer_consistency"):
        IndexedPreferenceDataset(root=str(tmp_path), tokenizer_consistency="warng")


def test_single_parquet_numpy_array_cells(tmp_path):
    # Parquet list-columns often round-trip as numpy arrays, not python lists.
    path = str(tmp_path / "pairs.parquet")
    pd.DataFrame(
        {
            "prompt_input_ids": [np.array([1, 2, 3], dtype=np.int64)],
            "chosen_input_ids": [np.array([6, 7], dtype=np.int64)],
            "rejected_input_ids": [np.array([8], dtype=np.int64)],
        }
    ).to_parquet(path)
    ds = SingleParquetPreferenceDataset(parquet_path=path)
    ex = ds[0]
    assert ex.prompt_input_ids == [1, 2, 3]
    assert ex.chosen_input_ids == [6, 7]
    assert ex.rejected_input_ids == [8]
    assert all(isinstance(t, int) for t in ex.chosen_input_ids)


def test_single_parquet_drop_empty_and_max_samples(tmp_path):
    # Exercises SingleParquetPreferenceDataset's is_empty predicate + max_samples cap.
    path = str(tmp_path / "pairs.parquet")
    pd.DataFrame(
        {
            "prompt_input_ids": [[1, 2], [3, 4], [5, 6]],
            "chosen_input_ids": [[7], [], [8, 9]],      # row 1 has an empty chosen -> dropped
            "rejected_input_ids": [[10], [11], [12]],
        }
    ).to_parquet(path)
    ds = SingleParquetPreferenceDataset(parquet_path=path)
    assert len(ds) == 2  # row 1 dropped
    assert ds[0].prompt_input_ids == [1, 2]
    assert ds[1].prompt_input_ids == [5, 6]  # row 2, since row 1 was filtered out

    ds_capped = SingleParquetPreferenceDataset(parquet_path=path, max_samples=1)
    assert len(ds_capped) == 1


def test_build_offline_dataset_factory(tmp_path):
    _build_indexed_fixture(tmp_path)

    class _Cfg(dict):
        def get(self, k, d=None):
            return super().get(k, d)

    cfg = _Cfg(
        offpolicy_format="indexed",
        offpolicy_files=str(tmp_path),
        max_prompt_length=8,
        max_response_length=4,
    )
    ds = build_offline_dataset(cfg, tokenizer=None)
    assert isinstance(ds, IndexedPreferenceDataset)
    assert len(ds) == 2

    # single_parquet also routes via the registry/from_config.
    sp_path = str(tmp_path / "sp.parquet")
    pd.DataFrame(
        {
            "prompt_input_ids": [[1, 2]],
            "chosen_input_ids": [[3, 4]],
            "rejected_input_ids": [[5]],
        }
    ).to_parquet(sp_path)
    cfg_sp = _Cfg(offpolicy_format="single_parquet", offpolicy_files=sp_path)
    assert isinstance(build_offline_dataset(cfg_sp, tokenizer=None), SingleParquetPreferenceDataset)

    cfg_bad = _Cfg(offpolicy_format="nope", offpolicy_files=str(tmp_path))
    with pytest.raises(ValueError, match="Unknown data.offpolicy_format"):
        build_offline_dataset(cfg_bad, tokenizer=None)

    cfg_nopath = _Cfg(offpolicy_format="indexed")
    with pytest.raises(ValueError, match="offpolicy_files"):
        build_offline_dataset(cfg_nopath, tokenizer=None)


# ----------------------- collate + assembly (requires verl) -----------------------


def test_assemble_offpolicy_pairs_layout(tmp_path):
    pytest.importorskip("verl", reason="assembly path needs verl (DataProto/postprocess_data)")
    from spin.offline_data import (
        assemble_offpolicy_pairs,
        make_tokenized_offpolicy_collate_fn,
    )
    from verl import DataProto

    tok = _FakeTokenizer()
    max_prompt_length, max_response_length = 8, 4
    examples = [
        PreferenceExample([40, 41], [50, 51, 52, 53], [60, 61]),
        PreferenceExample([10, 11, 12], [20, 21], [30, 31, 32]),
    ]
    collate = make_tokenized_offpolicy_collate_fn(tok, max_prompt_length=max_prompt_length)
    batch_dict = collate(examples)
    batch = DataProto.from_single_dict(batch_dict)
    assert batch.batch.batch_size[0] == 2  # N prompts

    pairs = assemble_offpolicy_pairs(batch, tok, max_prompt_length, max_response_length)
    n = 2
    seq_len = max_prompt_length + max_response_length
    assert pairs.batch.batch_size[0] == 2 * n
    assert tuple(pairs.batch["input_ids"].shape) == (2 * n, seq_len)
    assert tuple(pairs.batch["responses"].shape) == (2 * n, max_response_length)
    assert tuple(pairs.batch["prompts"].shape) == (2 * n, max_prompt_length)

    # chosen rows [0..N-1], rejected rows [N..2N-1]
    resp = pairs.batch["responses"]
    assert resp[0].tolist() == [50, 51, 52, 53]               # chosen of example 0
    assert resp[2].tolist() == [60, 61, tok.pad_token_id, tok.pad_token_id]  # rejected of ex 0
    rmask = pairs.batch["response_mask"]
    assert rmask[2].tolist() == [1, 1, 0, 0]

    # prompt is left-padded to max_prompt_length
    prompts = pairs.batch["prompts"]
    assert prompts[0].tolist() == [tok.pad_token_id] * 6 + [40, 41]
