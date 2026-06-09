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
    ConcatPreferenceDataset,
    IndexedPreferenceDataset,
    SingleParquetPreferenceDataset,
    TextChatPreferenceDataset,
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


class _FakeChatTokenizer(_FakeTokenizer):
    """Deterministic chat tokenizer: chat template = 'role:content' joined by '|', with
    'add_generation_prompt' appending '|assistant:'; tokenization = per-character code.
    This makes the prompt text a strict prefix of the full text, so the response-suffix
    slicing in TextChatPreferenceDataset is exercised end-to-end."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        text = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "|assistant:"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class _Cfg(dict):
    """Minimal data-config stand-in supporting .get(key, default)."""

    def get(self, k, d=None):
        return super().get(k, d)


def _write_single_parquet(path, n, prompt_len=2, resp_len=2, base=0):
    pd.DataFrame(
        {
            "prompt_input_ids": [[base + i, base + i + 1][:prompt_len] for i in range(n)],
            "chosen_input_ids": [[1000 + i] * resp_len for i in range(n)],
            "rejected_input_ids": [[2000 + i] * resp_len for i in range(n)],
        }
    ).to_parquet(str(path))


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
    with pytest.raises(ValueError, match="Unknown offline format"):
        build_offline_dataset(cfg_bad, tokenizer=None)

    cfg_nopath = _Cfg(offpolicy_format="indexed")
    with pytest.raises(ValueError, match="offpolicy_files"):
        build_offline_dataset(cfg_nopath, tokenizer=None)


# --------------------- per-dataset selection / mixture (no verl) ---------------------


def test_select_rows_head_vs_random_and_determinism():
    sr = IndexedPreferenceDataset.select_rows  # staticmethod on the base class
    head = sr(10, None, False, 4, "t", selection="head")
    assert head.tolist() == [0, 1, 2, 3]

    r1 = sr(10, None, False, 4, "t", selection="random", seed=123)
    r2 = sr(10, None, False, 4, "t", selection="random", seed=123)
    assert len(r1) == 4
    assert r1.tolist() == r2.tolist()                      # deterministic given seed
    assert r1.tolist() == sorted(r1.tolist())              # sorted for read locality
    assert set(r1.tolist()).issubset(set(range(10)))
    # different seed -> (very likely) different subset
    r3 = sr(10, None, False, 4, "t", selection="random", seed=999)
    assert r1.tolist() != r3.tolist()
    # cap >= n keeps all; cap < 0 keeps all
    assert sr(3, None, False, 10, "t", selection="random", seed=1).tolist() == [0, 1, 2]
    assert sr(3, None, False, -1, "t").tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="unknown selection"):
        sr(5, None, False, 2, "t", selection="middle")


def test_indexed_random_selection_reproducible(tmp_path):
    # 4 distinct pairs; cap to 2 random, deterministic by seed.
    accepted = [[i, i + 1, 100 + i, 101 + i] for i in range(4)]
    rejected = [[i, i + 1, 200 + i] for i in range(4)]
    _write_megatron_indexed(str(tmp_path / "accepted"), accepted)
    _write_megatron_indexed(str(tmp_path / "rejected"), rejected)
    pd.DataFrame({"chosen_index": list(range(4)), "rejected_index": list(range(4))}).to_parquet(
        str(tmp_path / "pairs.parquet")
    )
    a = IndexedPreferenceDataset(root=str(tmp_path), max_samples=2, selection="random", seed=7)
    b = IndexedPreferenceDataset(root=str(tmp_path), max_samples=2, selection="random", seed=7)
    assert len(a) == 2
    assert [a[i].prompt_input_ids for i in range(2)] == [b[i].prompt_input_ids for i in range(2)]


def test_concat_preference_dataset(tmp_path):
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _write_single_parquet(p1, 3, base=0)
    _write_single_parquet(p2, 2, base=500)
    d1 = SingleParquetPreferenceDataset(parquet_path=str(p1))
    d2 = SingleParquetPreferenceDataset(parquet_path=str(p2))
    concat = ConcatPreferenceDataset([d1, d2])
    assert len(concat) == 5
    assert concat[0].prompt_input_ids == d1[0].prompt_input_ids        # first source
    assert concat[3].prompt_input_ids == d2[0].prompt_input_ids        # crosses into 2nd
    assert concat[4].prompt_input_ids == d2[1].prompt_input_ids
    with pytest.raises(ValueError, match="at least one"):
        ConcatPreferenceDataset([])


def test_build_mixture_per_dataset_max_samples(tmp_path):
    # Full dataset 1 (5 rows) + 2 random rows from dataset 2 (10 rows) -> pool of 7.
    p1, p2 = tmp_path / "d1.parquet", tmp_path / "d2.parquet"
    _write_single_parquet(p1, 5, base=0)
    _write_single_parquet(p2, 10, base=500)
    cfg = _Cfg(
        offpolicy_format="single_parquet",
        offpolicy_datasets=[
            {"path": str(p1)},                                    # full
            {"path": str(p2), "max_samples": 2, "selection": "random"},
        ],
        seed=0,
    )
    ds = build_offline_dataset(cfg, tokenizer=None)
    assert isinstance(ds, ConcatPreferenceDataset)
    assert len(ds) == 7  # 5 + 2
    with pytest.raises(ValueError, match="missing required key 'path'"):
        build_offline_dataset(_Cfg(offpolicy_format="single_parquet", offpolicy_datasets=[{}]), tokenizer=None)


def test_text_chat_dataset(tmp_path):
    path = str(tmp_path / "text.parquet")
    pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "hi"}]],
            "chosen_response": [{"role": "assistant", "content": "hello"}],
            "rejected_response": [{"role": "assistant", "content": "no"}],
        }
    ).to_parquet(path)
    ds = TextChatPreferenceDataset(parquet_path=path, tokenizer=_FakeChatTokenizer())
    assert len(ds) == 1
    ex = ds[0]
    # prompt text "user:hi|assistant:" is a prefix of full; response = the suffix chars.
    assert ex.chosen_input_ids == [ord(c) for c in "hello"]
    assert ex.rejected_input_ids == [ord(c) for c in "no"]
    assert ex.prompt_input_ids == [ord(c) for c in "user:hi|assistant:"]


def test_build_mixture_text_and_tokenized(tmp_path):
    # Mix a pre-tokenized single_parquet dataset with a text_chat dataset in one stream.
    sp = tmp_path / "sp.parquet"
    _write_single_parquet(sp, 3, base=0)
    txt = str(tmp_path / "text.parquet")
    pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "q"}], [{"role": "user", "content": "w"}]],
            "chosen_response": [{"role": "assistant", "content": "aa"}, {"role": "assistant", "content": "bb"}],
            "rejected_response": [{"role": "assistant", "content": "x"}, {"role": "assistant", "content": "y"}],
        }
    ).to_parquet(txt)
    cfg = _Cfg(
        offpolicy_format="single_parquet",  # default for entries without explicit format
        offpolicy_datasets=[
            {"path": str(sp)},
            {"path": txt, "format": "text_chat"},
        ],
        seed=0,
    )
    ds = build_offline_dataset(cfg, tokenizer=_FakeChatTokenizer())
    assert isinstance(ds, ConcatPreferenceDataset)
    assert len(ds) == 5  # 3 tokenized + 2 text
    # last two come from the text dataset
    assert ds[3].chosen_input_ids == [ord(c) for c in "aa"]
    assert ds[4].rejected_input_ids == [ord(c) for c in "y"]


def test_text_chat_as_messages_normalization():
    msgs = [{"role": "user", "content": "x"}]
    assert TextChatPreferenceDataset._as_messages(list(msgs)) == msgs
    assert TextChatPreferenceDataset._as_messages(np.array(msgs, dtype=object)) == msgs


def test_text_chat_drop_empty(tmp_path):
    path = str(tmp_path / "t.parquet")
    pd.DataFrame(
        {
            "prompt": [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ],
            # row 1 has an empty chosen response -> dropped by drop_empty
            "chosen_response": [{"role": "assistant", "content": "x"}, {"role": "assistant", "content": "  "}],
            "rejected_response": [{"role": "assistant", "content": "p"}, {"role": "assistant", "content": "q"}],
        }
    ).to_parquet(path)
    ds = TextChatPreferenceDataset(parquet_path=path, tokenizer=_FakeChatTokenizer())
    assert len(ds) == 1
    assert ds[0].chosen_input_ids == [ord(c) for c in "x"]


def test_text_chat_random_selection_deterministic(tmp_path):
    path = str(tmp_path / "t.parquet")
    pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": str(i)}] for i in range(6)],
            "chosen_response": [{"role": "assistant", "content": f"c{i}"} for i in range(6)],
            "rejected_response": [{"role": "assistant", "content": f"r{i}"} for i in range(6)],
        }
    ).to_parquet(path)
    a = TextChatPreferenceDataset(path, _FakeChatTokenizer(), max_samples=3, selection="random", seed=5)
    b = TextChatPreferenceDataset(path, _FakeChatTokenizer(), max_samples=3, selection="random", seed=5)
    assert len(a) == 3
    assert [a[i].chosen_input_ids for i in range(3)] == [b[i].chosen_input_ids for i in range(3)]


def test_build_mixture_takes_precedence_over_files(tmp_path):
    # When both offpolicy_files and offpolicy_datasets are set, the mixture wins.
    p1 = tmp_path / "d1.parquet"
    _write_single_parquet(p1, 2)
    ignored = tmp_path / "ignored.parquet"
    _write_single_parquet(ignored, 9)
    cfg = _Cfg(
        offpolicy_format="single_parquet",
        offpolicy_files=str(ignored),
        offpolicy_datasets=[{"path": str(p1)}],
    )
    ds = build_offline_dataset(cfg, tokenizer=None)
    assert isinstance(ds, ConcatPreferenceDataset)
    assert len(ds) == 2  # from the mixture (d1), not the 9-row offpolicy_files


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


def test_assemble_over_mixture(tmp_path):
    pytest.importorskip("verl", reason="assembly path needs verl (DataProto/postprocess_data)")
    from spin.offline_data import (
        assemble_offpolicy_pairs,
        make_tokenized_offpolicy_collate_fn,
    )
    from verl import DataProto

    # Pool a single_parquet dataset with a text_chat dataset, then collate+assemble the
    # whole mixture (the end-to-end path the trainer takes for a mixed offline stream).
    sp = tmp_path / "sp.parquet"
    _write_single_parquet(sp, 2, base=0, resp_len=2)
    txt = str(tmp_path / "t.parquet")
    pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "q"}]],
            "chosen_response": [{"role": "assistant", "content": "ab"}],
            "rejected_response": [{"role": "assistant", "content": "z"}],
        }
    ).to_parquet(txt)
    cfg = _Cfg(
        offpolicy_format="single_parquet",
        offpolicy_datasets=[{"path": str(sp)}, {"path": txt, "format": "text_chat"}],
        max_prompt_length=8,
        max_response_length=4,
        seed=0,
    )
    tok = _FakeChatTokenizer()
    ds = build_offline_dataset(cfg, tokenizer=tok)
    n = len(ds)
    assert n == 3  # 2 tokenized + 1 text

    collate = make_tokenized_offpolicy_collate_fn(tok, max_prompt_length=8)
    batch = DataProto.from_single_dict(collate([ds[i] for i in range(n)]))
    assert batch.batch.batch_size[0] == n

    pairs = assemble_offpolicy_pairs(batch, tok, 8, 4)
    assert pairs.batch.batch_size[0] == 2 * n
    assert tuple(pairs.batch["input_ids"].shape) == (2 * n, 8 + 4)
    # the text entry is the last row of each half; its chosen response is "ab"
    assert pairs.batch["responses"][n - 1].tolist()[:2] == [ord("a"), ord("b")]
