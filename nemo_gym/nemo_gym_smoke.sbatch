#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# NeMo Gym x verl smoke: math_with_judge env + token-exact SGLang model server
# (NVIDIA-NeMo/gym#1787) against verl's native SGLang rollout, GRPO, 2 nodes.
# Conventions copied from apertus/launch/multinode_async_sandbox/_verl_training.sbatch.

set -xeuo pipefail

while IFS='=' read -r name _; do
  case "${name}" in
    SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_*) unset "${name}" ;;
  esac
done < <(env)
ulimit -c 0

USERNAME="$(whoami)"
SCRATCH="/iopsstor/scratch/cscs/${USERNAME}"
export HOME="${SCRATCH}"
WORKING_DIR="${SCRATCH}/projects/verl"
export NEMO_GYM_ROOT="${SCRATCH}/NemoGym-sglang"
DATA_DIR="${SCRATCH}/nemo-gym/math_data"
MODEL="/capstor/store/cscs/swissai/infra01/reasoning/models/Apertus-1p5-8B-sft-capfilter-linear-it8816"
DEPS="${SCRATCH}/nemo_gym_pydeps"
PINS="${SCRATCH}/nemo_gym_pins"
RUN_DIR="${WORKING_DIR}/outputs/nemo-gym-smoke/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

ENVIRONMENT_PATH="/capstor/store/cscs/swissai/infra01/reasoning/raas/docker/vs:251215-degenstop/env.toml"
SRUN_ENV=(--environment="${ENVIRONMENT_PATH}" --container-writable)

export HF_HOME="${SCRATCH}/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKING_DIR}:${PINS}:${DEPS}"

nodes_array=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
head_node="${nodes_array[0]}"

# One-time shared install of nemo-gym (+uv for its per-server venvs) into a
# --target dir on scratch, visible to every node via PYTHONPATH.
if [ ! -f "${DEPS}/.ready" ]; then
  srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" bash -lc "
    pip install -q --target '${DEPS}' '${NEMO_GYM_ROOT}' uv && touch '${DEPS}/.ready'"
fi

head_node_ip="$(srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address | awk '{print $1}')"
port=6379
ip_head="${head_node_ip}:${port}"
export RAY_ADDRESS="${ip_head}"

srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -lc "cd '${WORKING_DIR}' && ray start --head --temp-dir=/tmp/ray-${USERNAME} --node-ip-address='${head_node_ip}' --port='${port}' --num-cpus 64 --num-gpus 4 --block" &
sleep 15
srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${nodes_array[1]}" \
  bash -lc "cd '${WORKING_DIR}' && ray start --temp-dir=/tmp/ray-${USERNAME} --address '${ip_head}' --num-cpus 64 --num-gpus 4 --block" &
sleep 20

srun --overlap "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -lc "cd '${WORKING_DIR}' && \
    NEMO_GYM_ROOT='${NEMO_GYM_ROOT}' \
    NEMO_GYM_PYDEPS='${PINS}' \
    PYTHONPATH='${WORKING_DIR}:${PINS}:${DEPS}' \
    TMPDIR=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/ti-${USERNAME} TRITON_CACHE_DIR=/tmp/tr-${USERNAME} \
    HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      algorithm.use_kl_in_reward=False \
      data.train_files='${DATA_DIR}/train.jsonl' \
      data.val_files='${DATA_DIR}/validation.jsonl' \
      +data.custom_cls.path='${WORKING_DIR}/recipe/nemo_gym/dataset.py' \
      +data.custom_cls.name=NeMoGymJSONLDataset \
      data.truncation=left \
      data.max_prompt_length=1024 \
      data.max_response_length=2048 \
      data.train_batch_size=8 \
      actor_rollout_ref.model.path='${MODEL}' \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=8 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.rollout.name=sglang \
      actor_rollout_ref.rollout.mode=async \
      actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
      actor_rollout_ref.rollout.n=4 \
      actor_rollout_ref.rollout.temperature=1.0 \
      +actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.nemo_gym.agent_loop.NeMoGymAgentLoopManager \
      actor_rollout_ref.rollout.agent.agent_loop_config_path='${WORKING_DIR}/recipe/nemo_gym/configs/math_sglang_apertus.yaml' \
      reward_model.reward_manager=dapo \
      +reward_model.reward_kwargs.max_resp_len=2048 \
      'trainer.logger=[console]' \
      trainer.project_name=nemo-gym-smoke \
      trainer.experiment_name=sglang-apertus8b-math \
      trainer.default_local_dir='${RUN_DIR}' \
      trainer.val_before_train=False \
      trainer.test_freq=-1 \
      trainer.save_freq=-1 \
      trainer.total_epochs=1 \
      trainer.total_training_steps=3 \
      trainer.nnodes=2 \
      trainer.n_gpus_per_node=4"
