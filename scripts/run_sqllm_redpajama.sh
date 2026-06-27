#!/usr/bin/env bash
set -euo pipefail
set -x

CUDA_DEVICE="${CUDA_DEVICE:-1}"
export GUIDEDQUANT_CUDA_DEVICE="$CUDA_DEVICE"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME=${1:-meta-llama/Llama-3.1-8B}
BITS=${2:-4}
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"

NUM_GROUPS_OPT=""
MODE_OPT=""
if [[ -n "${3:-}" && "$3" != "-m" ]]; then
  NUM_GROUPS_OPT="--num_groups $3"
  if [[ "${4:-}" == "-m" && -n "${5:-}" ]]; then
    MODE_OPT="--mode $5"
  fi
elif [[ "${3:-}" == "-m" && -n "${4:-}" ]]; then
  MODE_OPT="--mode $4"
fi

python quantize.py "$MODEL_NAME" \
  --seed_precision "$BITS" --parent_precision "$BITS" \
  --dataset redpajama --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  $NUM_GROUPS_OPT $MODE_OPT
