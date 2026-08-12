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
      Option A — vLLM 0.17 for the rollout path: MORE viable than first
      thought. Our own estate already runs custom vLLM images on this
      hardware (rob-poc: ghcr.io/robmsmt/vllm-cxi + swiss-ai/vllm_alps,
      vllm_apertus_1.5) — the "vLLM disabled as broken" note is baked into
      the Dec-2025 training image, possibly stale. Recipe is tested exactly
      on 0.17.0. TODO: ask Imanol/Matteo WHY vLLM was disabled in the image.
      Option B — SGLang: NOT from scratch. Upstream draft PR
      NVIDIA-NeMo/gym#1787 (Kh4L) is a token-exact SGLang model server that
      hit and fixed the precise failure we'd hit (prefix-stability assert +
      retokenization drift, 48/48 tool turns failing) and is
      convergence-validated full-scale (Qwen3-30B-A3B) on a fork branch.
      Also: #976 (open ask since Mar 2026, describes our exact situation),
      #1557 (second draft adaptor). Fastest path: trial #1787's branch;
      reviewing/testing it also buys goodwill upstream.
      Related: #2451/#2452 confirm the vLLM 0.17→0.25 pin pain is real and
      acknowledged — the vLLM path stays version-brittle either way.
- [ ] Adapt submit_math.sh -> Clariden sbatch (our launch.sh conventions,
      container env.toml, no verlai/verl docker image).
- [ ] First smoke: math env (configs/math.yaml), 6-node smoke shape,
      SANDBOX_BACKEND=none equivalent (no code data), gym servers co-located.
- [ ] Then: resources servers as rob-poc Deployments (stateless), measure
      rollout-loop latency vs co-located before committing.
