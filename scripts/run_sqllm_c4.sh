#!/usr/bin/env bash
set -x

MODEL_NAME=${1:-meta-llama/Llama-3.1-8B}
BITS=${2:-4}

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
  --dataset c4 --seq_len 4096 --num_examples 1024 \
  $NUM_GROUPS_OPT $MODE_OPT
