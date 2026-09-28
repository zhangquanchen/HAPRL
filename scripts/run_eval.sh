#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIO3_ROOT="${MINIO3_ROOT:-${CODE_ROOT}/third_party/Mini-o3}"

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${VAL_JSON:?Set VAL_JSON}"
: "${BASE_IMAGE_DIR:?Set BASE_IMAGE_DIR to the root used by relative image paths}"
: "${RUN_DIR:=${CODE_ROOT}/outputs/evaluation/$(date +%Y%m%d_%H%M%S)}"
: "${INTERNAL_OPENAI_BASE_URL:?Load the private judge environment}"
: "${INTERNAL_OPENAI_API_KEY:?Load the private judge environment}"
: "${INTERNAL_OPENAI_MODEL:?Load the private judge environment}"

resolve_from_root() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${CODE_ROOT}" "$1" ;;
  esac
}

MINIO3_ROOT="$(resolve_from_root "${MINIO3_ROOT}")"
MODEL_PATH="$(resolve_from_root "${MODEL_PATH}")"
VAL_JSON="$(resolve_from_root "${VAL_JSON}")"
BASE_IMAGE_DIR="$(resolve_from_root "${BASE_IMAGE_DIR}")"
RUN_DIR="$(resolve_from_root "${RUN_DIR}")"

for path in "${MINIO3_ROOT}" "${MODEL_PATH}" "${VAL_JSON}" "${BASE_IMAGE_DIR}"; do
  [[ -e "${path}" ]] || { echo "[fatal] missing required path: ${path}" >&2; exit 2; }
done

export PYTHONPATH="${MINIO3_ROOT}:${PYTHONPATH:-}"
export BASE_IMAGE_DIR
export QA_PROCESS_TOTAL_MODE="${QA_PROCESS_TOTAL_MODE:-acc}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_DISABLE_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0

REWARD_FN="${REWARD_FN:-qa_data}"
PROCESS_REWARD_WEIGHT="${PROCESS_REWARD_WEIGHT:-0.0}"
RESUME_MODE="${RESUME_MODE:-disable}"
TRAJ_DIR="${RUN_DIR}/trajectories"
mkdir -p "${TRAJ_DIR}"

cd "${MINIO3_ROOT}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.system_prompt="${SYSTEM_PROMPT_NAME:-tool_crop}" \
  "data.train_files=[${VAL_JSON}]" \
  "data.val_files=[${VAL_JSON}]" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-16}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-8192}" \
  data.image_key=images \
  data.answer_key=solution \
  data.mask_blank=False \
  data.acc_reward_weight=1.0 \
  data.format_reward_weight=0.0 \
  +data.process_reward_weight="${PROCESS_REWARD_WEIGHT}" \
  data.tool_call_penalty=0.0 \
  data.general_qa_reward_fn="${REWARD_FN}" \
  data.gpt_general_qa_reward_fn="${REWARD_FN}" \
  data.gpt_extract_answer=True \
  data.gpt_threads="${GPT_THREADS:-8}" \
  data.extract_answer_tags=strict \
  data.return_raw_chat=True \
  data.tool_call=crop \
  data.use_tgt_size=False \
  data.max_pixels="${MAX_PIXELS:-1000000}" \
  data.min_pixels="${MIN_PIXELS:-40000}" \
  reward_model.reward_manager=naive_multithreads_tool \
  reward_model.use_hybrid_reward_manager=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING:-False}" \
  actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.56}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-1}" \
  actor_rollout_ref.rollout.max_generation_round="${MAX_GENERATION_ROUND:-6}" \
  "actor_rollout_ref.rollout.limit_mm_per_prompt={image:6}" \
  actor_rollout_ref.rollout.val_max_generation_round="${MAX_GENERATION_ROUND:-6}" \
  "actor_rollout_ref.rollout.val_limit_mm_per_prompt={image:6}" \
  actor_rollout_ref.rollout.use_raw_image=True \
  actor_rollout_ref.rollout.use_relative_coordinates=True \
  actor_rollout_ref.rollout.multi_turn_prompt_type=v2 \
  actor_rollout_ref.rollout.vllm_infer_batch_size="${VLLM_INFER_BATCH_SIZE:-8}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  "actor_rollout_ref.rollout.stop=['</grounding>']" \
  actor_rollout_ref.rollout.save_traj_dir="${TRAJ_DIR}" \
  actor_rollout_ref.rollout.val_n="${VAL_N:-1}" \
  actor_rollout_ref.rollout.val_do_sample="${VAL_DO_SAMPLE:-False}" \
  trainer.default_local_dir="${CHECKPOINT_ROOT:-${RUN_DIR}/checkpoints}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.val_only=True \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.project_name=prm_release_eval \
  trainer.experiment_name="${EXPERIMENT_NAME:-evaluation}" \
  trainer.n_gpus_per_node="${N_GPUS:-8}" \
  trainer.num_workers=1 \
  trainer.nnodes=1 \
  trainer.save_freq=999999 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 \
  "$@"

