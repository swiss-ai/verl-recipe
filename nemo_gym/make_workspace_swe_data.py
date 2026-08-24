#!/usr/bin/env python3
"""Generate train/val jsonl for the workspace_swe env.

v0 = generated micro-repo bug-fix tasks. Each task ships a buggy file plus a
VISIBLE test (so the agent can run it, see the failure, and iterate — the
mid-episode learning signal) and a separate HELD-OUT hidden test used only at
<submit/> for the reward (so the model can't just hard-code the visible case).

    python3 make_workspace_swe_data.py /iopsstor/scratch/cscs/$USER/nemo-gym/workspace_swe_data
"""
import json
import os
import sys

SYSTEM = (
    "You are a software engineer in a code workspace. Each turn take exactly one "
    "action: <cmd>shell command</cmd> to run something, "
    '<write path="file">new contents</write> to edit a file, or <submit/> when done. '
    "Run the visible tests to check your work before submitting."
)

# (task, buggy_files_incl_visible_test, hidden_tests, test_cmd)
TASKS = [
    (
        "mathutils.add is wrong. Run the tests, fix mathutils.py, and submit.",
        {"mathutils.py": "def add(a, b):\n    return a - b  # BUG\n",
         "test_visible.py": "from mathutils import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"},
        {"hidden_tests/test_hidden.py": "from mathutils import add\n\n\ndef test_more():\n    assert add(-1, 1) == 0\n    assert add(10, 5) == 15\n"},
        "python -m pytest hidden_tests -q",
    ),
    (
        "strutils.reverse_words should reverse the word order. Fix it.",
        {"strutils.py": "def reverse_words(s):\n    return s  # BUG\n",
         "test_visible.py": "from strutils import reverse_words\n\n\ndef test_rev():\n    assert reverse_words('a b c') == 'c b a'\n"},
        {"hidden_tests/test_hidden.py": "from strutils import reverse_words\n\n\ndef test_more():\n    assert reverse_words('hello world') == 'world hello'\n    assert reverse_words('one') == 'one'\n"},
        "python -m pytest hidden_tests -q",
    ),
    (
        "fib(n) is off by one (fib(0)=0, fib(1)=1). Fix fib.py.",
        {"fib.py": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n + 1):  # BUG\n        a, b = b, a + b\n    return a\n",
         "test_visible.py": "from fib import fib\n\n\ndef test_small():\n    assert fib(0) == 0\n    assert fib(1) == 1\n"},
        {"hidden_tests/test_hidden.py": "from fib import fib\n\n\ndef test_seq():\n    assert [fib(i) for i in range(7)] == [0, 1, 1, 2, 3, 5, 8]\n"},
        "python -m pytest hidden_tests -q",
    ),
    (
        "dedupe must remove duplicates preserving order. Fix dedupe.py.",
        {"dedupe.py": "def dedupe(xs):\n    return list(set(xs))  # BUG: loses order\n",
         "test_visible.py": "from dedupe import dedupe\n\n\ndef test_basic():\n    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]\n"},
        {"hidden_tests/test_hidden.py": "from dedupe import dedupe\n\n\ndef test_more():\n    assert dedupe([1, 1, 1]) == [1]\n    assert dedupe([]) == []\n"},
        "python -m pytest hidden_tests -q",
    ),
]


def row(task, files, hidden, test_cmd):
    return {
        "responses_create_params": {"input": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Make the tests pass."},
        ]},
        "task": task, "files": files, "hidden_tests": hidden,
        "test_cmd": test_cmd, "max_steps": 12,
        "agent_ref": {"type": "responses_api_agents", "name": "workspace_swe_gymnasium_agent"},
    }


out_dir = sys.argv[1] if len(sys.argv) > 1 else "workspace_swe_data"
os.makedirs(out_dir, exist_ok=True)
rows = [row(*t) for t in TASKS]
train = [rows[i % len(rows)] for i in range(4096)]
val = list(rows)
with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
    f.write("\n".join(json.dumps(r) for r in train) + "\n")
with open(os.path.join(out_dir, "validation.jsonl"), "w") as f:
    f.write("\n".join(json.dumps(r) for r in val) + "\n")
print(f"wrote {out_dir}/train.jsonl ({len(train)} rows) and validation.jsonl ({len(val)} rows)")
