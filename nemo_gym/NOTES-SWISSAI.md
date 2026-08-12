# NeMo Gym x swiss-ai verl — integration branch notes (2026-08-12)

Vendored from verl-project/verl-recipe @ nemo_gym (folder commit
e1128e325f38b..., see REQUIRED_VERL.txt). Docs tutorial:
docs.nvidia.com/nemo/gym/main/tutorials/training-tutorials/verl

## How the integration works (per tutorial + recipe source)
- NeMo Gym servers (model server + resources servers) run as separate HTTP
  processes; verl talks to them during rollout via a custom agent loop.
- verl side = 3 hydra overrides, no core changes:
    +data.custom_cls.path=recipe/nemo_gym/dataset.py
    +data.custom_cls.name=NeMoGymJSONLDataset
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.nemo_gym.agent_loop.NeMoGymAgentLoopManager
    +actor_rollout_ref.rollout.agent.agent_loop_config_path=.../recipe/nemo_gym/configs/<env>.yaml
- Data: `gym dataset collate --resources-server <env>` -> train/val jsonl
  with agent_ref routing field (multi-env = mixed agent_ref in one dataset).
- Servers may be remote HTTP (rob-poc later) or cluster-local (tutorial uses
  SLURM co-location; start there).

## Verified on our infra already (Phase 1, 2026-08-12)
- nemo-gym runs on Clariden aarch64 (uv, py3.13); reasoning_gym env e2e green
  against CSCS-served Apertus-v1.5-8B (4/4 rollouts, mean reward 0.25).
- Working eval command + secrets-safe pattern: ~/work/eth/RL/nemo-gym-standup.md

## Compat checklist before first training smoke
- [x] **Core API: COMPATIBLE (verified 2026-08-12).** The fork's
      verl/experimental/agent_loop is at upstream parity for every symbol the
      recipe imports: AgentLoopManager, AgentLoopMetrics,
      _InternalAgentLoopOutput, AgentLoopWorker._postprocess,
      distillation_enabled, reward_loop_worker_handles — identical grep counts
      fork vs up/main. agent_loop_manager_class config hook present (4 refs
      both sides). vllm_async_server module exists in the fork tree.
- [x] **server_patch.py read: it is vLLM-0.17-specific** — monkeypatches
      vllm.entrypoints OpenAIServingChat + OpenAIServingTokenization so token
      IDs/logprobs stay consistent between NeMo Gym's OpenAI-style calls and
      the trainer (prevents retokenization mismatch). replica.py likewise
      subclasses vLLMHttpServer/vLLMReplica.
- [ ] **DECISION NEEDED — the one real gap: recipe rollout path is vLLM-only;
      our stack is SGLang (vLLM disabled as broken in the CSCS image).**
      Option A: revive vLLM rollout just for this recipe (fights the image).
      Option B (recommended): write NeMoGymSGLangReplica against the fork's
      async SGLang server + port the token-fidelity logic; SGLang exposes
      token ids/logprobs natively (skip_tokenizer_init, return_logprob), so
      the patch may shrink or vanish. Scope: ~days, not weeks.
- [ ] Adapt submit_math.sh -> Clariden sbatch (our launch.sh conventions,
      container env.toml, no verlai/verl docker image).
- [ ] First smoke: math env (configs/math.yaml), 6-node smoke shape,
      SANDBOX_BACKEND=none equivalent (no code data), gym servers co-located.
- [ ] Then: resources servers as rob-poc Deployments (stateless), measure
      rollout-loop latency vs co-located before committing.
