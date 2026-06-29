#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[rbvt-sqllm] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
MODEL_BASENAME="${MODEL_NAME##*/}"
BITS="${BITS:-3}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
CACHE_DIR="${CACHE_DIR:-cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CALIB_SEED="${CALIB_SEED:-0}"
MIRROR_DATASET="${MIRROR_DATASET:-ZengXiangyu/RedPajama-Data-1T-Sample}"

RBVT_N_CALIB="${RBVT_N_CALIB:-1024}"
RBVT_BATCH_SIZE="${RBVT_BATCH_SIZE:-4}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
RBVT_ROW_CHUNK="${RBVT_ROW_CHUNK:-4096}"
RBVT_GAP_FLOOR="${RBVT_GAP_FLOOR:-1e-8}"

REPO_EVAL_CONTEXT="${REPO_EVAL_CONTEXT:-8192}"
NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-2048}"
NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"

RUN_SQLLM="${RUN_SQLLM:-1}"
RUN_RBVT="${RUN_RBVT:-1}"
RUN_REPO_EVAL="${RUN_REPO_EVAL:-1}"
RUN_NONUQ_EVAL="${RUN_NONUQ_EVAL:-1}"
OVERWRITE_RBVT="${OVERWRITE_RBVT:-0}"
OVERWRITE_RBVT_STATS="${OVERWRITE_RBVT_STATS:-0}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ "${PYTHON_BIN}" == "python" ]] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
fi

TOKEN_PATH="${CACHE_DIR}/tokens/${MODEL_BASENAME}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"
SQLLM_QUANTIZED_PATH="${CACHE_DIR}/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
SQLLM_PACKED_PATH="${CACHE_DIR}/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
RBVT_PACKED_PATH="${CACHE_DIR}/rbvt_sqllm_packed/anyprec-rbvt-sqllm-${MODEL_BASENAME}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}"
RBVT_STATS_PATH="${CACHE_DIR}/rbvt_sqllm_stats/${MODEL_BASENAME}-w${BITS}-rbvt-sqllm-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}_n${RBVT_N_CALIB}.pt"
REPO_RESULTS_FILE="results_rbvt_sqllm_${MODEL_BASENAME}_redpajama${SEQ_LEN}_${BITS}bit_ctx${REPO_EVAL_CONTEXT}.json"
NONUQ_RESULTS_FILE="outputs/nonuquant_ppl_rbvt_sqllm_${MODEL_BASENAME}_${BITS}bit.json"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [rbvt-sqllm] $*"
}

cleanup_cuda() {
  "${PYTHON_BIN}" - <<'PY' || true
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception:
    pass
PY
}

ensure_mirror_calibration() {
  if [[ -s "${TOKEN_PATH}" ]]; then
    log "Using existing RedPajama mirror calibration tokens: ${TOKEN_PATH}"
    return
  fi

  log "Creating ${NUM_EXAMPLES}x${SEQ_LEN} RedPajama calibration tokens from ${MIRROR_DATASET}"
  mkdir -p "$(dirname "${TOKEN_PATH}")"
  "${PYTHON_BIN}" - "${MODEL_NAME}" "${MIRROR_DATASET}" "${TOKEN_PATH}" "${SEQ_LEN}" "${NUM_EXAMPLES}" "${CALIB_SEED}" <<'PY'
import os
import random
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

model_name, dataset_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
seq_len, num_samples, seed = map(int, sys.argv[4:7])
random.seed(seed)

tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
ds = load_dataset(dataset_name, split="train")
indices = list(range(len(ds)))
random.shuffle(indices)

samples = []
pbar = tqdm(total=num_samples, desc=f"Tokenizing {dataset_name}")
for idx in indices:
    ids = tok(ds[idx]["text"], return_tensors="pt").input_ids[0]
    if ids.numel() < seq_len:
        continue
    start = random.randint(0, ids.numel() - seq_len)
    samples.append(ids[start:start + seq_len].long())
    pbar.update(1)
    if len(samples) == num_samples:
        break
pbar.close()

if len(samples) != num_samples:
    raise SystemExit(f"Only collected {len(samples)} / {num_samples} calibration samples")

tokens = torch.stack(samples)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(tokens, out_path)
print(f"saved {out_path} shape={tuple(tokens.shape)}")
PY
}

run_squeezellm_if_needed() {
  if [[ "${RUN_SQLLM}" != "1" ]]; then
    log "RUN_SQLLM=0; skipping SqueezeLLM base quantization"
    return
  fi

  if [[ -d "${SQLLM_QUANTIZED_PATH}" && -d "${SQLLM_PACKED_PATH}" ]]; then
    log "Using existing SqueezeLLM cache: ${SQLLM_QUANTIZED_PATH}"
    return
  fi

  log "Running GuidedQuant SqueezeLLM base quantization"
  "${PYTHON_BIN}" quantize.py "${MODEL_NAME}" \
    --seed_precision "${BITS}" \
    --parent_precision "${BITS}" \
    --cache_dir "${CACHE_DIR}" \
    --dataset redpajama \
    --seq_len "${SEQ_LEN}" \
    --num_examples "${NUM_EXAMPLES}"
  cleanup_cuda
}

run_rbvt_squeezellm() {
  if [[ "${RUN_RBVT}" != "1" ]]; then
    log "RUN_RBVT=0; skipping RBVT"
    return
  fi

  local overwrite_args=()
  if [[ "${OVERWRITE_RBVT}" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi
  if [[ "${OVERWRITE_RBVT_STATS}" == "1" ]]; then
    overwrite_args+=(--overwrite-stats)
  fi

  log "Running RBVT on GuidedQuant SqueezeLLM assignments"
  "${PYTHON_BIN}" rbvt_squeezellm.py \
    --model "${MODEL_NAME}" \
    --bits "${BITS}" \
    --cache-dir "${CACHE_DIR}" \
    --dataset redpajama \
    --seq-len "${SEQ_LEN}" \
    --num-examples "${NUM_EXAMPLES}" \
    --tokens-path "${TOKEN_PATH}" \
    --input-quantized-path "${SQLLM_QUANTIZED_PATH}" \
    --output-packed-path "${RBVT_PACKED_PATH}" \
    --stats-path "${RBVT_STATS_PATH}" \
    --n-calib "${RBVT_N_CALIB}" \
    --batch-size "${RBVT_BATCH_SIZE}" \
    --rbvt-lambda "${RBVT_LAMBDA}" \
    --rbvt-topk "${RBVT_TOPK}" \
    --row-chunk "${RBVT_ROW_CHUNK}" \
    --gap-floor "${RBVT_GAP_FLOOR}" \
    "${overwrite_args[@]}"
  cleanup_cuda
}

run_repo_eval_targeted() {
  if [[ "${RUN_REPO_EVAL}" != "1" ]]; then
    log "RUN_REPO_EVAL=0; skipping repo-style eval"
    return
  fi

  log "Running targeted GuidedQuant repo-style PPL eval, context=${REPO_EVAL_CONTEXT}"
  "${PYTHON_BIN}" - "${RBVT_PACKED_PATH}" "${REPO_EVAL_CONTEXT}" "${REPO_RESULTS_FILE}" <<'PY'
import gc
import json
import sys

import torch
from any_precision.evaluate import eval as ap_eval

model_path, context, output_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
name = model_path.rstrip("/").split("/")[-1]
tokenizer_type, tokenizer, model = ap_eval.auto_model_load(model_path, verbose=True)
ppl = ap_eval.evaluate_ppl(
    model=model,
    tokenizer=tokenizer,
    testcases=["wikitext2", "c4"],
    verbose=True,
    chunk_size=context,
    tokenizer_type=tokenizer_type,
)
payload = {name: {"ppl": ppl}}
with open(output_file, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
print(json.dumps(payload, indent=2, sort_keys=True))
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
PY
  cleanup_cuda
}

run_nonuq_eval() {
  if [[ "${RUN_NONUQ_EVAL}" != "1" ]]; then
    log "RUN_NONUQ_EVAL=0; skipping NonUQuantFix-style eval"
    return
  fi

  log "Running NonUQuantFix-style sliding-window PPL eval"
  mkdir -p outputs
  "${PYTHON_BIN}" scripts/eval_nonuquant_style_ppl.py \
    --model-path "${RBVT_PACKED_PATH}" \
    --model-name "rbvt_sqllm_${MODEL_BASENAME}" \
    --tokenizer-path "${MODEL_NAME}" \
    --datasets wikitext2 c4 \
    --precision "${BITS}" \
    --max-length "${NONUQ_MAX_LENGTH}" \
    --stride "${NONUQ_STRIDE}" \
    --c4-samples "${NONUQ_C4_SAMPLES}" \
    --output-file "${NONUQ_RESULTS_FILE}"
}

main() {
  log "Target model: ${MODEL_NAME}"
  log "Calibration: ${MIRROR_DATASET}, ${NUM_EXAMPLES}x${SEQ_LEN}, seed=${CALIB_SEED}"
  log "RBVT config: lambda=${RBVT_LAMBDA}, topk=${RBVT_TOPK}, row_chunk=${RBVT_ROW_CHUNK}, batch=${RBVT_BATCH_SIZE}, n_calib=${RBVT_N_CALIB}"
  ensure_mirror_calibration
  run_squeezellm_if_needed
  run_rbvt_squeezellm
  run_repo_eval_targeted
  run_nonuq_eval
  log "Done. Packed model: ${RBVT_PACKED_PATH}"
  log "Repo eval: ${REPO_RESULTS_FILE}"
  log "NonUQuant eval: ${NONUQ_RESULTS_FILE}"
}

main "$@"
