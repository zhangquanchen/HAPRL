# HaPRL: Human-Anchored Process Reinforcement Learning for Visual Search Agent

Annotations and training/evaluation code are included. You still need to supply:

1. base model weights in `models/base/`
2. benchmark images under `data/mydata/data/`
3. an OpenAI-compatible vision judge in `.private/judge.env`

## Quickstart

Linux, Python 3.11, CUDA 12.x, and a multi-GPU node with BF16. SFT and RL
**must** use two separate conda environments.

```bash
export CODE_ROOT="$(pwd)"
export MINIO3_ROOT="$CODE_ROOT/third_party/Mini-o3"
export BASE_IMAGE_DIR="$CODE_ROOT/data/mydata"
export OUTPUT_ROOT="$CODE_ROOT/outputs"

# 1) Check the bundled JSON (no GPU, no images required)
python3 data/verify_data.py

# 2) Put images so relative paths resolve, then:
#    data/mydata/data/VisualProbe_train_repo/image/<file>.jpg
#    data/mydata/data/visualprobe_medium/image/<file>.jpg
#    data/mydata/data/VisualProbe_Easy_repo/image/<file>.jpg
#    data/mydata/data/VisualProbe_Hard_repo/image/<file>.jpg
python3 data/verify_data.py --check-images

# 3) Put the base model, for example:
#    models/base/Qwen3-VL-8B-Instruct

# 4) SFT environment (do not mix with RL)
conda create -n prm-sft python=3.11 -y
conda activate prm-sft
bash scripts/setup_env.sh sft
llamafactory-cli train configs/sft.yaml

# 5) RL environment (new interpreter)
conda create -n prm-rl python=3.11 -y
conda activate prm-rl
bash scripts/setup_env.sh rl

mkdir -p .private
cp environment/judge.env.example .private/judge.env
chmod 600 .private/judge.env
# edit .private/judge.env: endpoint, API key, vision model name

set -a
source configs/rl.env.example
source .private/judge.env
set +a
export MODEL_PATH="$CODE_ROOT/models/checkpoints/sft_start"

# 6) RL: answer-gated process reward
QA_PROCESS_TOTAL_MODE=process bash scripts/run_rl.sh

# matched answer-only control
QA_PROCESS_TOTAL_MODE=acc bash scripts/run_rl.sh

# 7) Evaluation
export MODEL_PATH=models/checkpoints/model_to_evaluate
export VAL_JSON=data/mydata/qa_data/test.json
export RUN_DIR=outputs/evaluation/run_name/qa_test/greedy
bash scripts/run_eval.sh
```

Default RL files: `data/mydata/train_final/train1000_v1.json` (train) and
`data/mydata/qa_process/val_mini_acc.json` (val). Hyperparameters live in
`configs/rl.env.example`.

## What you need

| Item | Where |
| --- | --- |
| Annotations | `data/mydata/` (already in the package) |
| Images | `data/mydata/data/<repo>/image/` (download from the original benchmarks) |
| Base model | `models/base/` then set `MODEL_PATH` |
| Vision judge | copy `environment/judge.env.example` to `.private/judge.env` |
| SFT | `prm-sft`: Python 3.11, PyTorch 2.7, LLaMA-Factory 0.9.5, DeepSpeed 0.19 |
| RL / eval | `prm-rl`: Python 3.11, PyTorch 2.8, Transformers 4.57, vLLM 0.11, Ray 2.55 |

`BASE_IMAGE_DIR` must be `data/mydata`, because JSON paths look like
`./data/VisualProbe_train_repo/image/<file>.jpg`.

## Method (short)

The policy answers visual questions with a crop tool, up to six rounds, then
emits a final answer. Training is SFT then GRPO. The default RL objective is:

```text
process_reward = answer_correctness × weighted_process_score
```

Process score uses five dimensions: `target_semantics`,
`evidence_acquisition`, `search_progress`, `tool_discipline`,
`communication_discipline`. Set `QA_PROCESS_TOTAL_MODE=acc` to optimize
answer correctness only; keep every other setting identical.

`scripts/run_rl.sh` already sets GRPO, `naive_multithreads_tool`,
`vllm_multi_turn_tool_call`, `tool_crop`, multi-turn prompt `v2`, and
`qa_process_v1`. Confirm both a text and an image judge request from a
compute node before a full run.

## Data

| File | Role |
| --- | --- |
| `data/mydata/train_final/train1000_v1.json` | main RL train (`qa_process_v1`) |
| `data/mydata/train_final/train1000.json` | same samples, earlier `qa_process` schema |
| `data/mydata/train_final/train.json` | earlier 600-sample merge |
| `data/mydata/qa_process/` | QA splits with process supervision |
| `data/mydata/qa_data/` | same QA splits, answer only |
| `data/mydata/iou_process/` | grounding splits with process supervision |
| `data/mydata/iou_data/` | grounding splits, answer only |

`qa_process/val_mini_acc.json` is the default RL validation set (answer-only
scoring). `_process` vs `_data` is the same samples with or without process
fields. Process text is derived from human GUI trajectories, then rewritten
into `Round N` form; the raw GUI exports are not in this package.

Evaluation uses the same crop tool and six-round limit. Deterministic eval:
`VAL_N=1`, `ROLLOUT_N=1`, `VAL_DO_SAMPLE=False`. Do not mix it with
multi-sample runs. Grounding answers look like
`<answer><bbox>x1,y1,x2,y2</bbox></answer>` with coordinates in `[0, 1]`.

## Checkpoints

Resume needs trainer state, actor shards, optimizer state, and model shards.
Weights-only checkpoints are for inference only. Always write a new output
directory; do not overwrite the source checkpoint.
