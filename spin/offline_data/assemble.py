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

"""Token-id assembly for offline preference pairs.

``assemble_offpolicy_pairs`` is the drop-in, tokenizer-free twin of
``spin_trainer.tokenize_offpolicy_pairs``: given a collated batch that already carries
left-padded prompt ``input_ids`` plus ragged chosen/rejected RESPONSE token-id lists,
it builds the exact same ``2N``-row DataProto (chosen rows ``[0..N-1]``, rejected rows
``[N..2N-1]``) — but performs NO ``apply_chat_template`` / ``tokenizer()`` calls.

Output tensor layout (identical to ``tokenize_offpolicy_pairs``)::

    input_ids       [2N, prompt_len + max_response_length]  long, prompt left-padded
    attention_mask  [2N, prompt_len + max_response_length]  long (0 = pad)
    position_ids    [2N, prompt_len + max_response_length]  long
    responses       [2N, max_response_length]               long, response right-padded
    response_mask   [2N, max_response_length]               long (0 = pad)
    prompts         [2N, prompt_len]                         long
"""

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from .base import CHOSEN_RESPONSE_IDS_KEY, REJECTED_RESPONSE_IDS_KEY


def assemble_offpolicy_pairs(
    batch: "DataProto", tokenizer, max_prompt_length: int, max_response_length: int
) -> "DataProto":
    """Build the off-policy chosen/rejected DataProto from PRE-TOKENIZED ids.

    Mirrors ``tokenize_offpolicy_pairs`` exactly (signature, ordering, padding) so the
    trainer's downstream off-policy block needs no changes; only the source of the
    response token ids differs (carried as non-tensors instead of re-tokenized text).
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_input_ids = batch.batch["input_ids"]  # [N, prompt_len] (left-padded)
    prompt_attention_mask = batch.batch["attention_mask"]  # [N, prompt_len]
    n = prompt_input_ids.shape[0]

    chosen_responses = batch.non_tensor_batch[CHOSEN_RESPONSE_IDS_KEY]  # array of int lists
    rejected_responses = batch.non_tensor_batch[REJECTED_RESPONSE_IDS_KEY]

    all_input_ids = []
    all_attention_mask = []
    all_responses = []
    all_response_mask = []
    all_prompts = []

    for side_responses in [chosen_responses, rejected_responses]:
        for i in range(n):
            p_ids = prompt_input_ids[i]  # [prompt_len]
            p_mask = prompt_attention_mask[i]  # [prompt_len]

            resp_ids = list(side_responses[i])
            if len(resp_ids) > max_response_length:
                resp_ids = resp_ids[:max_response_length]
            resp_len_actual = len(resp_ids)

            resp_tensor = torch.full((max_response_length,), pad_id, dtype=torch.long)
            if resp_len_actual > 0:
                resp_tensor[:resp_len_actual] = torch.tensor(resp_ids, dtype=torch.long)
            resp_mask = torch.zeros(max_response_length, dtype=torch.long)
            resp_mask[:resp_len_actual] = 1

            seq_ids = torch.cat([p_ids, resp_tensor], dim=0)
            seq_mask = torch.cat([p_mask, resp_mask], dim=0)

            all_input_ids.append(seq_ids)
            all_attention_mask.append(seq_mask)
            all_responses.append(resp_tensor)
            all_response_mask.append(resp_mask)
            all_prompts.append(p_ids)

    input_ids = torch.stack(all_input_ids, dim=0)
    attention_mask = torch.stack(all_attention_mask, dim=0)
    position_ids = compute_position_id_with_mask(attention_mask)
    responses = torch.stack(all_responses, dim=0)
    response_mask = torch.stack(all_response_mask, dim=0)
    prompts = torch.stack(all_prompts, dim=0)

    td = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "prompts": prompts,
        },
        batch_size=2 * n,
    )
    return DataProto(batch=td)
