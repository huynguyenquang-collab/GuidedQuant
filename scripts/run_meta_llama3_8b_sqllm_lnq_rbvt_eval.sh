#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[llama3-8b-job] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
MODEL_BASENAME="${MODEL_NAME##*/}"
BITS="${BITS:-3}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
CACHE_DIR="${CACHE_DIR:-cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LNQ_NUM_GROUPS="${LNQ_NUM_GROUPS:-1}"
LNQ_NUM_ITERATIONS="${LNQ_NUM_ITERATIONS:-2}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"

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
NONUQ_BATCH_SIZE="${NONUQ_BATCH_SIZE:-4}"
NONUQ_EVAL_MODE="${NONUQ_EVAL_MODE:-sliding}"

CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-1}"
RUN_QUANT="${RUN_QUANT:-1}"
RUN_RBVT="${RUN_RBVT:-1}"
RUN_REPO_EVAL="${RUN_REPO_EVAL:-1}"
RUN_NONUQ_EVAL="${RUN_NONUQ_EVAL:-1}"
OVERWRITE_RBVT_STATS="${OVERWRITE_RBVT_STATS:-1}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export GUIDEDQUANT_CACHE_DEQUANT="${GUIDEDQUANT_CACHE_DEQUANT:-1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ "${PYTHON_BIN}" == "python" ]] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
fi

TOKEN_URL="${TOKEN_URL:-https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Meta-Llama-3-8B-redpajama_s1024_blk4096.pt}"
TOKEN_PATH="${CACHE_DIR}/tokens/${MODEL_BASENAME}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"

SQLLM_QUANTIZED_PATH="${CACHE_DIR}/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
SQLLM_PACKED_PATH="${CACHE_DIR}/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}"

LNQ_QUANTIZED_PATH="${CACHE_DIR}/layerwise_quantized/${MODEL_BASENAME}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"
LNQ_PACKED_PATH="${CACHE_DIR}/layerwise_packed/layerwise-${MODEL_BASENAME}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"

RBVT_QUANTIZED_PATH="${CACHE_DIR}/rbvt_sqllm_quantized/${MODEL_BASENAME}-w${BITS}-rbvt-sqllm-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}"
RBVT_PACKED_PATH="${CACHE_DIR}/rbvt_sqllm_packed/anyprec-rbvt-sqllm-${MODEL_BASENAME}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}"
RBVT_STATS_PATH="${CACHE_DIR}/rbvt_sqllm_stats/${MODEL_BASENAME}-w${BITS}-rbvt-sqllm-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}_n${RBVT_N_CALIB}.pt"

REPO_RESULTS_FILE="results_meta_llama3_8b_sqllm_lnq_rbvt_redpajama${SEQ_LEN}_${BITS}bit_ctx${REPO_EVAL_CONTEXT}.json"
NONUQ_OUTPUT_DIR="outputs/meta_llama3_8b_sqllm_lnq_rbvt_nonuq"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [llama3-8b-job] $*"
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

ensure_official_calibration() {
  if [[ -s "${TOKEN_PATH}" ]]; then
    log "Using existing official calibration tokens: ${TOKEN_PATH}"
    return
  fi

  mkdir -p "$(dirname "${TOKEN_PATH}")"
  log "Downloading official calibration tokens: ${TOKEN_URL}"
  local tmp_path="${TOKEN_PATH}.tmp"
  rm -f "${tmp_path}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${tmp_path}" "${TOKEN_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${TOKEN_URL}" -o "${tmp_path}"
  else
    echo "Neither wget nor curl is installed; cannot download calibration tokens." >&2
    exit 1
  fi
  mv "${tmp_path}" "${TOKEN_PATH}"
}

verify_calibration() {
  "${PYTHON_BIN}" - "${TOKEN_PATH}" <<'PY'
import sys
import torch

path = sys.argv[1]
tokens = torch.load(path, map_location="cpu")
if isinstance(tokens, torch.Tensor):
    shape = tuple(tokens.shape)
elif isinstance(tokens, (list, tuple)):
    shape = (len(tokens), tuple(tokens[0].shape) if tokens else None)
else:
    raise SystemExit(f"Unsupported calibration type: {type(tokens).__name__}")
print(f"calibration={path} type={type(tokens).__name__} shape={shape}")
PY
}

clean_outputs() {
  if [[ "${CLEAN_OUTPUTS}" != "1" ]]; then
    log "CLEAN_OUTPUTS=0; keeping existing generated outputs"
    return
  fi
  log "Removing generated outputs for this Meta-Llama-3-8B job"
  rm -rf \
    "${SQLLM_QUANTIZED_PATH}" \
    "${SQLLM_PACKED_PATH}" \
    "${LNQ_QUANTIZED_PATH}" \
    "${LNQ_PACKED_PATH}" \
    "${RBVT_QUANTIZED_PATH}" \
    "${RBVT_PACKED_PATH}" \
    "${RBVT_STATS_PATH}" \
    "${REPO_RESULTS_FILE}" \
    "${NONUQ_OUTPUT_DIR}"
}

run_squeezellm() {
  log "Running SqueezeLLM: ${MODEL_NAME}, ${BITS}-bit, RedPajama ${NUM_EXAMPLES}x${SEQ_LEN}"
  "${PYTHON_BIN}" quantize.py "${MODEL_NAME}" \
    --seed_precision "${BITS}" \
    --parent_precision "${BITS}" \
    --cache_dir "${CACHE_DIR}" \
    --dataset redpajama \
    --seq_len "${SEQ_LEN}" \
    --num_examples "${NUM_EXAMPLES}" \
    --overwrite_quantize \
    --overwrite_pack
  cleanup_cuda
}

run_lnq_plain() {
  log "Running plain LNQ: ${MODEL_NAME}, ${BITS}-bit, g=${LNQ_NUM_GROUPS}, T=${LNQ_NUM_ITERATIONS}, K=${LNQ_CD_CYCLES}"
  "${PYTHON_BIN}" layerwise_nuq.py "${MODEL_NAME}" \
    --seed_precision "${BITS}" \
    --cache_dir "${CACHE_DIR}" \
    --dataset redpajama \
    --seq_len "${SEQ_LEN}" \
    --num_examples "${NUM_EXAMPLES}" \
    --num_groups "${LNQ_NUM_GROUPS}" \
    --num_iterations "${LNQ_NUM_ITERATIONS}" \
    --cd_cycles "${LNQ_CD_CYCLES}" \
    --is_nosal true \
    --overwrite_quantize \
    --overwrite_pack
  cleanup_cuda
}

run_rbvt_squeezellm() {
  log "Running RBVT-SqueezeLLM: lambda=${RBVT_LAMBDA}, topk=${RBVT_TOPK}, row_chunk=${RBVT_ROW_CHUNK}, batch=${RBVT_BATCH_SIZE}, n_calib=${RBVT_N_CALIB}"
  local overwrite_stats_args=()
  if [[ "${OVERWRITE_RBVT_STATS}" == "1" ]]; then
    overwrite_stats_args+=(--overwrite-stats)
  fi
  "${PYTHON_BIN}" rbvt_squeezellm.py \
    --model "${MODEL_NAME}" \
    --bits "${BITS}" \
    --cache-dir "${CACHE_DIR}" \
    --dataset redpajama \
    --seq-len "${SEQ_LEN}" \
    --num-examples "${NUM_EXAMPLES}" \
    --tokens-path "${TOKEN_PATH}" \
    --input-quantized-path "${SQLLM_QUANTIZED_PATH}" \
    --output-quantized-path "${RBVT_QUANTIZED_PATH}" \
    --output-packed-path "${RBVT_PACKED_PATH}" \
    --stats-path "${RBVT_STATS_PATH}" \
    --n-calib "${RBVT_N_CALIB}" \
    --batch-size "${RBVT_BATCH_SIZE}" \
    --rbvt-lambda "${RBVT_LAMBDA}" \
    --rbvt-topk "${RBVT_TOPK}" \
    --row-chunk "${RBVT_ROW_CHUNK}" \
    --gap-floor "${RBVT_GAP_FLOOR}" \
    --overwrite \
    "${overwrite_stats_args[@]}"
  cleanup_cuda
}

run_repo_eval() {
  if [[ "${RUN_REPO_EVAL}" != "1" ]]; then
    log "RUN_REPO_EVAL=0; skipping repo-style eval"
    return
  fi
  log "Running targeted repo-style PPL eval, context=${REPO_EVAL_CONTEXT}"
  "${PYTHON_BIN}" - "${REPO_EVAL_CONTEXT}" "${REPO_RESULTS_FILE}" "${SQLLM_PACKED_PATH}" "${LNQ_PACKED_PATH}" "${RBVT_PACKED_PATH}" <<'PY'
import gc
import json
import sys

import torch
from any_precision.evaluate import eval as ap_eval

context = int(sys.argv[1])
output_file = sys.argv[2]
model_paths = sys.argv[3:]
payload = {}

for model_path in model_paths:
    name = model_path.rstrip("/").split("/")[-1]
    print(f"\n===== repo-style eval: {name} | ctx={context} =====")
    tokenizer_type, tokenizer, model = ap_eval.auto_model_load(model_path, verbose=True)
    ppl = ap_eval.evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        testcases=["wikitext2", "c4"],
        verbose=True,
        chunk_size=context,
        tokenizer_type=tokenizer_type,
    )
    payload[name] = {"ppl": ppl}
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

with open(output_file, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  cleanup_cuda
}

run_nonuq_eval() {
  if [[ "${RUN_NONUQ_EVAL}" != "1" ]]; then
    log "RUN_NONUQ_EVAL=0; skipping NonUQuantFix-style eval"
    return
  fi

  log "Running NonUQuantFix-style sliding-window PPL eval"
  mkdir -p "${NONUQ_OUTPUT_DIR}"
  "${PYTHON_BIN}" scripts/eval_nonuquant_style_ppl.py \
    --model-path "${SQLLM_PACKED_PATH}" \
    --model-name "squeezellm_${MODEL_BASENAME}" \
    --tokenizer-path "${MODEL_NAME}" \
    --datasets wikitext2 c4 \
    --precision "${BITS}" \
    --max-length "${NONUQ_MAX_LENGTH}" \
    --stride "${NONUQ_STRIDE}" \
    --batch-size "${NONUQ_BATCH_SIZE}" \
    --eval-mode "${NONUQ_EVAL_MODE}" \
    --c4-samples "${NONUQ_C4_SAMPLES}" \
    --output-file "${NONUQ_OUTPUT_DIR}/squeezellm.json"
  cleanup_cuda

  "${PYTHON_BIN}" scripts/eval_nonuquant_style_ppl.py \
    --model-path "${LNQ_PACKED_PATH}" \
    --model-name "lnq_plain_${MODEL_BASENAME}" \
    --tokenizer-path "${MODEL_NAME}" \
    --datasets wikitext2 c4 \
    --precision "${BITS}" \
    --max-length "${NONUQ_MAX_LENGTH}" \
    --stride "${NONUQ_STRIDE}" \
    --batch-size "${NONUQ_BATCH_SIZE}" \
    --eval-mode "${NONUQ_EVAL_MODE}" \
    --c4-samples "${NONUQ_C4_SAMPLES}" \
    --output-file "${NONUQ_OUTPUT_DIR}/lnq_plain.json"
  cleanup_cuda

  "${PYTHON_BIN}" scripts/eval_nonuquant_style_ppl.py \
    --model-path "${RBVT_PACKED_PATH}" \
    --model-name "rbvt_sqllm_${MODEL_BASENAME}" \
    --tokenizer-path "${MODEL_NAME}" \
    --datasets wikitext2 c4 \
    --precision "${BITS}" \
    --max-length "${NONUQ_MAX_LENGTH}" \
    --stride "${NONUQ_STRIDE}" \
    --batch-size "${NONUQ_BATCH_SIZE}" \
    --eval-mode "${NONUQ_EVAL_MODE}" \
    --c4-samples "${NONUQ_C4_SAMPLES}" \
    --output-file "${NONUQ_OUTPUT_DIR}/rbvt_squeezellm.json"
}

main() {
  log "Target model: ${MODEL_NAME}"
  log "Calibration: official ${MODEL_BASENAME} RedPajama ${NUM_EXAMPLES}x${SEQ_LEN}"
  log "Repo eval context: ${REPO_EVAL_CONTEXT}; NonUQuant eval: max_length=${NONUQ_MAX_LENGTH}, stride=${NONUQ_STRIDE}"
  ensure_official_calibration
  verify_calibration
  clean_outputs

  if [[ "${RUN_QUANT}" == "1" ]]; then
    run_squeezellm
    run_lnq_plain
  else
    log "RUN_QUANT=0; using existing SqueezeLLM/LNQ artifacts"
  fi

  if [[ "${RUN_RBVT}" == "1" ]]; then
    run_rbvt_squeezellm
  else
    log "RUN_RBVT=0; using existing RBVT artifact"
  fi

  run_repo_eval
  run_nonuq_eval

  log "Done."
  log "SqueezeLLM: ${SQLLM_PACKED_PATH}"
  log "LNQ plain: ${LNQ_PACKED_PATH}"
  log "RBVT-SqueezeLLM: ${RBVT_PACKED_PATH}"
  log "Repo eval: ${REPO_RESULTS_FILE}"
  log "NonUQuant eval dir: ${NONUQ_OUTPUT_DIR}"
}

main "$@"
