#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIO3_ROOT="${MINIO3_ROOT:-${CODE_ROOT}/third_party/Mini-o3}"

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${TRAIN_JSON:?Set TRAIN_JSON}"
: "${VAL_JSON:?Set VAL_JSON}"
: "${BASE_IMAGE_DIR:?Set BASE_IMAGE_DIR to the root used by relative image paths}"
: "${OUTPUT_ROOT:=${CODE_ROOT}/outputs/rl}"
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
TRAIN_JSON="$(resolve_from_root "${TRAIN_JSON}")"
VAL_JSON="$(resolve_from_root "${VAL_JSON}")"
BASE_IMAGE_DIR="$(resolve_from_root "${BASE_IMAGE_DIR}")"
OUTPUT_ROOT="$(resolve_from_root "${OUTPUT_ROOT}")"

for path in "${MINIO3_ROOT}" "${MODEL_PATH}" "${TRAIN_JSON}" "${VAL_JSON}" "${BASE_IMAGE_DIR}"; do
  [[ -e "${path}" ]] || { echo "[fatal] missing required path: ${path}" >&2; exit 2; }
done

export PYTHONPATH="${MINIO3_ROOT}:${PYTHONPATH:-}"
export BASE_IMAGE_DIR
export QA_PROCESS_TOTAL_MODE="${QA_PROCESS_TOTAL_MODE:-process}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_DISABLE_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0

RUN_NAME="${RUN_NAME:-rl_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
CKPT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"

cd "${MINIO3_ROOT}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.system_prompt=tool_crop \
  "data.train_files=[${TRAIN_JSON}]" \
  "data.val_files=[${VAL_JSON}]" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-48}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-8192}" \
  data.image_key=images \
  data.answer_key=solution \
  data.mask_blank=False \
  data.acc_reward_weight=1.0 \
  data.format_reward_weight=0.0 \
  +data.process_reward_weight="${PROCESS_REWARD_WEIGHT:-1.0}" \
  data.tool_call_penalty=0.0 \
  data.general_qa_reward_fn=qa_process_v1 \
  data.gpt_general_qa_reward_fn=qa_process_v1 \
  data.gpt_extract_answer=True \
  data.gpt_threads="${GPT_THREADS:-24}" \
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
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-5e-7}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.003 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_multi_turn_response_mask=True \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-1.0}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.62}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.max_generation_round="${MAX_GENERATION_ROUND:-6}" \
  "actor_rollout_ref.rollout.limit_mm_per_prompt={image:6}" \
  actor_rollout_ref.rollout.val_max_generation_round="${MAX_GENERATION_ROUND:-6}" \
  "actor_rollout_ref.rollout.val_limit_mm_per_prompt={image:6}" \
  actor_rollout_ref.rollout.use_raw_image=True \
  actor_rollout_ref.rollout.use_relative_coordinates=True \
  actor_rollout_ref.rollout.multi_turn_prompt_type=v2 \
  actor_rollout_ref.rollout.vllm_infer_batch_size="${VLLM_INFER_BATCH_SIZE:-16}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  "actor_rollout_ref.rollout.stop=['</grounding>']" \
  trainer.rejection_sample=True \
  trainer.rejection_sample_multiplier=1 \
  "trainer.logger=['console','tensorboard']" \
  trainer.project_name=prm_release \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS:-8}" \
  trainer.num_workers=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-23}" \
  trainer.test_freq="${TEST_FREQ:-6}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-5}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  "$@"

