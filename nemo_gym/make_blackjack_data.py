#!/usr/bin/env python3
"""Generate train/val jsonl for the blackjack env.

Every episode is randomized server-side, so rows are identical prompts; the
dataset's only job is to set the batch count and name the agent. Usage:

    python3 make_blackjack_data.py /iopsstor/scratch/cscs/$USER/nemo-gym/blackjack_data
"""
import json
import os
import sys

ROW = {
    "responses_create_params": {
        "input": [
            {
                "role": "system",
                "content": (
                    "You are playing Blackjack. After seeing your hand, respond with "
                    "<action>hit</action> or <action>stand</action>. Think briefly, "
                    "then give your action tag."
                ),
            },
            {"role": "user", "content": "Deal me in."},
        ]
    },
    "agent_ref": {"type": "responses_api_agents", "name": "blackjack_gymnasium_agent"},
}

out_dir = sys.argv[1] if len(sys.argv) > 1 else "blackjack_data"
os.makedirs(out_dir, exist_ok=True)
line = json.dumps(ROW)
with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
    f.write("\n".join([line] * 4096) + "\n")
with open(os.path.join(out_dir, "validation.jsonl"), "w") as f:
    f.write("\n".join([line] * 64) + "\n")
print(f"wrote {out_dir}/train.jsonl (4096 rows) and validation.jsonl (64 rows)")
