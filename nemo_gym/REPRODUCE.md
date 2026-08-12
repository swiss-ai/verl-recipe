# Reproduce: NeMo Gym × verl × SGLang training smoke on Clariden

Green run: job 3069020 (2026-08-12) — 3 GRPO steps, 2×4 GH200,
Apertus-1.5-8B, rewards 0.031 → 0.156 → 0.188. ~9 min end to end.
Chain: swiss-ai verl (1p5-async-rl) → native SGLang rollout (mode=async) →
NeMoGymAgentLoopManager (this dir) → token-exact sglang_model server
(NVIDIA-NeMo/gym PR #1787) → math_with_judge env.

## Prerequisites
- Clariden account in `infra01` (reservation `SD-69241-apertus-1-5-0`, ends Aug 31)
- `HF_TOKEN` exported in your shell (dataset collate needs it)
- An **aarch64** `uv` binary on PATH (`file $(which uv)` must say ARM — an
  x86 uv fails inside server spawns with "Exec format error")

## One-time setup (paths assume your iopsstor scratch = $S)

```bash
S=/iopsstor/scratch/cscs/$USER

# 1. NeMo Gym fork with the token-exact SGLang server (pin the SHA — fork
#    branches can rebase) + lower its ray floor to match the container:
git clone -b sglang-splice-fix https://github.com/Kh4L/NemoGym $S/NemoGym-sglang
cd $S/NemoGym-sglang && git checkout 71f2243be6bc
sed -i 's/ray\[default\]>=2.55.1/ray[default]>=2.52.1/' pyproject.toml
#    (their 2.55 floor is CVE-driven; the container already runs 2.52 — the
#    real remediation is a container image bump, tracked separately)

# 2. verl fork + this recipe (recipe/ is a git submodule in verl, so we
#    drop the files in rather than init the submodule):
git clone -b 1p5-async-rl https://github.com/swiss-ai/verl $S/projects/verl
git clone -b feat/nemo-gym-envs https://github.com/swiss-ai/verl-recipe /tmp/vr
mkdir -p $S/projects/verl/recipe && touch $S/projects/verl/recipe/__init__.py
cp -r /tmp/vr/nemo_gym $S/projects/verl/recipe/

# 3. Vendored deps (TWO dirs — see NOTES-SWISSAI.md for why split matters):
#    builds $S/nemo_gym_pydeps + $S/nemo_gym_pins, probes the import chain
#    inside the container, touches .ready on success:
sed "s|/iopsstor/scratch/cscs/rosmith|$S|g" \
  $S/projects/verl/recipe/nemo_gym/setup_nemo_deps.sh > $S/setup_nemo_deps.sh
srun --account=infra01 --reservation=SD-69241-apertus-1-5-0 --partition=normal \
  --nodes=1 --ntasks=1 --time=00:30:00 --container-writable \
  --environment=/capstor/store/cscs/swissai/infra01/reasoning/raas/docker/vs:251215-degenstop/env.toml \
  bash $S/setup_nemo_deps.sh
# expect: "vendored pins verified" ... "probe[N]: IMPORT_OK" ... "DONE"

# 4. Training data (math_with_judge -> train/validation.jsonl with agent_ref):
cd $S/NemoGym-sglang && uv run --python 3.13 --with-editable . \
  gym dataset collate --resources-server math_with_judge \
  --model-type openai_model --mode train_preparation \
  --output-dir $S/nemo-gym/math_data --download "+hf_token=$HF_TOKEN"
# (run via srun or the xfer partition, not a login node)
```

## Run the smoke

```bash
sed "s|/iopsstor/scratch/cscs/rosmith|$S|g" \
  $S/projects/verl/recipe/nemo_gym/nemo_gym_smoke.sbatch > $S/nemo_gym_smoke.sbatch
cd $S/projects/verl/outputs
sbatch --job-name=nemo-gym-smoke $S/nemo_gym_smoke.sbatch
tail -f nemo-gym-smoke_<jobid>.err   # progress bars live here
```

Expected timeline: ~1 min prologue → ~5 min model load → 4× "Uvicorn
running" (gym servers; first run builds venvs, +3 min) → "Collecting
rollouts" bars → 3 training steps (~15–50 s each) → COMPLETED at ~9 min.
Step metrics (`critic/rewards/mean`) are in the `.out` file.

## Gotchas we already hit so you don't have to
- **Stale server venvs**: if a run dies during venv resolution, `rm -rf`
  every `.venv` under the NemoGym checkout before retrying — half-built
  venvs get treated as ready and servers crash on missing imports.
- **pip --target** needs `--ignore-installed --upgrade`, one package per
  invocation (silently skips/collides otherwise).
- **Container overlays don't persist** across srun steps (CE, not pyxis) —
  never rely on in-container pip installs surviving to the next step.
- **nemo_gym rewrites sys.path at import** and demotes its own install dir —
  that's why the anthropic/openai pins live in a separate `nemo_gym_pins`.
- OOM in dataloader → keep `--mem=0` on sruns and
  `data.dataloader_num_workers=0` (both already in the sbatch).

## What to change for real experiments
- `trainer.total_training_steps`, `data.train_batch_size`,
  `actor_rollout_ref.rollout.n` in the sbatch.
- Env: swap `configs/math_sglang_apertus.yaml` config_paths to another
  resources server; recollate data accordingly. Multi-env = one dataset
  with mixed `agent_ref` values (see README.rst / submit_multienv.sh).
- Scale shape: nnodes + trainer.nnodes + rollout TP as usual.
