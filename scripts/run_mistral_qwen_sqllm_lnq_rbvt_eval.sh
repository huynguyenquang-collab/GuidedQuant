#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[mistral-qwen-job] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Llama-2-style reproduction setup for additional 7B-class models:
# - RedPajama calibration, 1024 samples x 4096 tokens
# - 3-bit SqueezeLLM, plain LNQ (--is_nosal true), and RBVT-SqueezeLLM
# - PPL eval on WikiText2 and C4 using both repo-style and NonUQuant-style paths
declare -A MODEL_ALIASES=(
  [Mistral7Bv03]="mistralai/Mistral-7B-v0.3"
  [Qwen25_7B]="Qwen/Qwen2.5-7B"
  [Qwen3_8B]="Qwen/Qwen3-8B"
  [Qwen35_9B]="Qwen/Qwen3.5-9B"
)

MODELS="${MODELS:-Mistral7Bv03 Qwen25_7B}"
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

REPO_EVAL_CONTEXT="${REPO_EVAL_CONTEXT:-4096}"
NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-2048}"
NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"

CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-1}"
RUN_QUANT="${RUN_QUANT:-1}"
RUN_RBVT="${RUN_RBVT:-1}"
RUN_REPO_EVAL="${RUN_REPO_EVAL:-1}"
RUN_NONUQ_EVAL="${RUN_NONUQ_EVAL:-1}"
OVERWRITE_RBVT_STATS="${OVERWRITE_RBVT_STATS:-1}"

# If no model-specific calibration token exists locally and no official token URL
# is configured, build a reproducible cache from a RedPajama sample dataset.
# Default to the mirror because the official Data-1T loader has unstable metadata
# schemas across shards. Override these vars to use the official dataset again.
OFFICIAL_REDPJ_DATASET="${OFFICIAL_REDPJ_DATASET:-ZengXiangyu/RedPajama-Data-1T-Sample}"
OFFICIAL_REDPJ_CONFIG="${OFFICIAL_REDPJ_CONFIG:-}"
OFFICIAL_REDPJ_SPLIT="${OFFICIAL_REDPJ_SPLIT:-train}"
OFFICIAL_REDPJ_TEXT_FIELD="${OFFICIAL_REDPJ_TEXT_FIELD:-text}"
OFFICIAL_REDPJ_RANDOM_STATE="${OFFICIAL_REDPJ_RANDOM_STATE:-0}"
OFFICIAL_REDPJ_SHUFFLE_BUFFER="${OFFICIAL_REDPJ_SHUFFLE_BUFFER:-10000}"
TRUST_EXISTING_CALIB="${TRUST_EXISTING_CALIB:-1}"

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

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [mistral-qwen-job] $*"
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

resolve_model() {
  local model="$1"
  if [[ -n "${MODEL_ALIASES[$model]+set}" ]]; then
    echo "${MODEL_ALIASES[$model]}"
  else
    echo "${model}"
  fi
}

model_token_url() {
  local basename="$1"
  local env_name
  env_name="TOKEN_URL_${basename//[^A-Za-z0-9_]/_}"
  echo "${!env_name:-}"
}

verify_calibration() {
  local token_path="$1"
  "${PYTHON_BIN}" - "${token_path}" <<'PY'
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

calibration_source_string() {
  printf 'redpajama_calib dataset=%s config=%s split=%s text_field=%s seed=%s shuffle_buffer=%s seq_len=%s num_examples=%s\n' \
    "${OFFICIAL_REDPJ_DATASET}" \
    "${OFFICIAL_REDPJ_CONFIG}" \
    "${OFFICIAL_REDPJ_SPLIT}" \
    "${OFFICIAL_REDPJ_TEXT_FIELD}" \
    "${OFFICIAL_REDPJ_RANDOM_STATE}" \
    "${OFFICIAL_REDPJ_SHUFFLE_BUFFER}" \
    "${SEQ_LEN}" \
    "${NUM_EXAMPLES}"
}

calibration_source_matches() {
  local token_path="$1"
  local source_path="${token_path}.source"
  [[ -s "${token_path}" && -f "${source_path}" ]] || return 1
  [[ "$(cat "${source_path}")" == "$(calibration_source_string)" ]]
}

download_file() {
  local url="$1"
  local output="$2"
  local tmp_path="${output}.tmp"

  mkdir -p "$(dirname "${output}")"
  rm -f "${tmp_path}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${tmp_path}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${tmp_path}"
  else
    echo "Neither wget nor curl is installed; cannot download calibration tokens." >&2
    exit 1
  fi
  mv "${tmp_path}" "${output}"
}

build_official_redpajama_calibration() {
  local model_name="$1"
  local token_path="$2"

  log "Building calibration tokens from ${OFFICIAL_REDPJ_DATASET}${OFFICIAL_REDPJ_CONFIG:+/${OFFICIAL_REDPJ_CONFIG}}"
  local calib_args=(
    scripts/make_official_redpajama_calib.py
    --model "${model_name}" \
    --output "${token_path}" \
    --seq-len "${SEQ_LEN}" \
    --num-examples "${NUM_EXAMPLES}" \
    --dataset "${OFFICIAL_REDPJ_DATASET}" \
    --split "${OFFICIAL_REDPJ_SPLIT}" \
    --text-field "${OFFICIAL_REDPJ_TEXT_FIELD}" \
    --seed "${OFFICIAL_REDPJ_RANDOM_STATE}" \
    --shuffle-buffer "${OFFICIAL_REDPJ_SHUFFLE_BUFFER}"
  )
  if [[ -n "${OFFICIAL_REDPJ_CONFIG}" ]]; then
    calib_args+=(--config "${OFFICIAL_REDPJ_CONFIG}")
  fi
  "${PYTHON_BIN}" "${calib_args[@]}"
  calibration_source_string > "${token_path}.source"
}

ensure_calibration() {
  local model_name="$1"
  local basename="$2"
  local token_path="$3"
  local token_url

  if calibration_source_matches "${token_path}"; then
    log "Using existing RedPajama calibration tokens: ${token_path}"
    return
  fi

  if [[ -s "${token_path}" && "${TRUST_EXISTING_CALIB}" == "1" ]]; then
    log "TRUST_EXISTING_CALIB=1; using existing calibration tokens without source marker: ${token_path}"
    return
  fi

  if [[ -s "${token_path}" ]]; then
    log "Existing calibration cache lacks matching source marker; rebuilding: ${token_path}"
    rm -f "${token_path}" "${token_path}.source"
  fi

  token_url="$(model_token_url "${basename}")"
  if [[ -n "${token_url}" ]]; then
    log "Downloading model-specific calibration tokens: ${token_url}"
    download_file "${token_url}" "${token_path}"
    printf 'downloaded url=%s seq_len=%s num_examples=%s\n' "${token_url}" "${SEQ_LEN}" "${NUM_EXAMPLES}" > "${token_path}.source"
    return
  fi

  mkdir -p "$(dirname "${token_path}")"
  build_official_redpajama_calibration "${model_name}" "${token_path}"
}

clean_outputs() {
  local paths=("$@")
  if [[ "${CLEAN_OUTPUTS}" != "1" ]]; then
    log "CLEAN_OUTPUTS=0; keeping existing generated outputs"
    return
  fi
  log "Removing generated outputs for this model"
  rm -rf "${paths[@]}"
}

run_squeezellm() {
  local model_name="$1"
  log "Running SqueezeLLM: ${model_name}, ${BITS}-bit, RedPajama ${NUM_EXAMPLES}x${SEQ_LEN}"
  "${PYTHON_BIN}" quantize.py "${model_name}" \
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
  local model_name="$1"
  log "Running plain LNQ: ${model_name}, ${BITS}-bit, g=${LNQ_NUM_GROUPS}, T=${LNQ_NUM_ITERATIONS}, K=${LNQ_CD_CYCLES}"
  "${PYTHON_BIN}" layerwise_nuq.py "${model_name}" \
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
  local model_name="$1"
  local token_path="$2"
  local sqllm_quantized_path="$3"
  local rbvt_quantized_path="$4"
  local rbvt_packed_path="$5"
  local rbvt_stats_path="$6"

  log "Running RBVT-SqueezeLLM: lambda=${RBVT_LAMBDA}, topk=${RBVT_TOPK}, row_chunk=${RBVT_ROW_CHUNK}, batch=${RBVT_BATCH_SIZE}, n_calib=${RBVT_N_CALIB}"
  local overwrite_stats_args=()
  if [[ "${OVERWRITE_RBVT_STATS}" == "1" ]]; then
    overwrite_stats_args+=(--overwrite-stats)
  fi
  "${PYTHON_BIN}" rbvt_squeezellm.py \
    --model "${model_name}" \
    --bits "${BITS}" \
    --cache-dir "${CACHE_DIR}" \
    --dataset redpajama \
    --seq-len "${SEQ_LEN}" \
    --num-examples "${NUM_EXAMPLES}" \
    --tokens-path "${token_path}" \
    --input-quantized-path "${sqllm_quantized_path}" \
    --output-quantized-path "${rbvt_quantized_path}" \
    --output-packed-path "${rbvt_packed_path}" \
    --stats-path "${rbvt_stats_path}" \
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
  local output_file="$1"
  shift
  local model_paths=("$@")

  if [[ "${RUN_REPO_EVAL}" != "1" ]]; then
    log "RUN_REPO_EVAL=0; skipping repo-style eval"
    return
  fi

  log "Running targeted repo-style PPL eval, context=${REPO_EVAL_CONTEXT}"
  "${PYTHON_BIN}" - "${REPO_EVAL_CONTEXT}" "${output_file}" "${model_paths[@]}" <<'PY'
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

run_nonuq_eval_one() {
  local model_path="$1"
  local model_name="$2"
  local tokenizer_path="$3"
  local output_file="$4"

  "${PYTHON_BIN}" scripts/eval_nonuquant_style_ppl.py \
    --model-path "${model_path}" \
    --model-name "${model_name}" \
    --tokenizer-path "${tokenizer_path}" \
    --datasets wikitext2 c4 \
    --precision "${BITS}" \
    --max-length "${NONUQ_MAX_LENGTH}" \
    --stride "${NONUQ_STRIDE}" \
    --c4-samples "${NONUQ_C4_SAMPLES}" \
    --output-file "${output_file}"
  cleanup_cuda
}

run_nonuq_eval() {
  local model_name="$1"
  local basename="$2"
  local output_dir="$3"
  local sqllm_packed_path="$4"
  local lnq_packed_path="$5"
  local rbvt_packed_path="$6"

  if [[ "${RUN_NONUQ_EVAL}" != "1" ]]; then
    log "RUN_NONUQ_EVAL=0; skipping NonUQuantFix-style eval"
    return
  fi

  log "Running NonUQuantFix-style sliding-window PPL eval"
  mkdir -p "${output_dir}"
  run_nonuq_eval_one "${sqllm_packed_path}" "squeezellm_${basename}" "${model_name}" "${output_dir}/squeezellm.json"
  run_nonuq_eval_one "${lnq_packed_path}" "lnq_plain_${basename}" "${model_name}" "${output_dir}/lnq_plain.json"
  run_nonuq_eval_one "${rbvt_packed_path}" "rbvt_sqllm_${basename}" "${model_name}" "${output_dir}/rbvt_squeezellm.json"
}

run_model_job() {
  local requested="$1"
  local model_name
  local basename

  model_name="$(resolve_model "${requested}")"
  basename="${model_name##*/}"

  local token_path="${CACHE_DIR}/tokens/${basename}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"
  local sqllm_quantized_path="${CACHE_DIR}/quantized/${basename}-w${BITS}_orig${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
  local sqllm_packed_path="${CACHE_DIR}/packed/anyprec-${basename}-w${BITS}_orig${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
  local lnq_quantized_path="${CACHE_DIR}/layerwise_quantized/${basename}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"
  local lnq_packed_path="${CACHE_DIR}/layerwise_packed/layerwise-${basename}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"
  local rbvt_quantized_path="${CACHE_DIR}/rbvt_sqllm_quantized/${basename}-w${BITS}-rbvt-sqllm-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}"
  local rbvt_packed_path="${CACHE_DIR}/rbvt_sqllm_packed/anyprec-rbvt-sqllm-${basename}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}"
  local rbvt_stats_path="${CACHE_DIR}/rbvt_sqllm_stats/${basename}-w${BITS}-rbvt-sqllm-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}_n${RBVT_N_CALIB}.pt"
  local repo_results_file="results_${basename}_sqllm_lnq_rbvt_redpajama${SEQ_LEN}_${BITS}bit_ctx${REPO_EVAL_CONTEXT}.json"
  local nonuq_output_dir="outputs/${basename}_sqllm_lnq_rbvt_nonuq"

  log "============================================================"
  log "Target model: ${model_name}"
  log "Calibration: ${NUM_EXAMPLES}x${SEQ_LEN}; token path ${token_path}"
  log "Repo eval context: ${REPO_EVAL_CONTEXT}; NonUQuant eval: max_length=${NONUQ_MAX_LENGTH}, stride=${NONUQ_STRIDE}"

  ensure_calibration "${model_name}" "${basename}" "${token_path}"
  verify_calibration "${token_path}"
  clean_outputs \
    "${sqllm_quantized_path}" \
    "${sqllm_packed_path}" \
    "${lnq_quantized_path}" \
    "${lnq_packed_path}" \
    "${rbvt_quantized_path}" \
    "${rbvt_packed_path}" \
    "${rbvt_stats_path}" \
    "${repo_results_file}" \
    "${nonuq_output_dir}"

  if [[ "${RUN_QUANT}" == "1" ]]; then
    run_squeezellm "${model_name}"
    run_lnq_plain "${model_name}"
  else
    log "RUN_QUANT=0; using existing SqueezeLLM/LNQ artifacts"
  fi

  if [[ "${RUN_RBVT}" == "1" ]]; then
    run_rbvt_squeezellm "${model_name}" "${token_path}" "${sqllm_quantized_path}" "${rbvt_quantized_path}" "${rbvt_packed_path}" "${rbvt_stats_path}"
  else
    log "RUN_RBVT=0; using existing RBVT artifact"
  fi

  run_repo_eval "${repo_results_file}" "${sqllm_packed_path}" "${lnq_packed_path}" "${rbvt_packed_path}"
  run_nonuq_eval "${model_name}" "${basename}" "${nonuq_output_dir}" "${sqllm_packed_path}" "${lnq_packed_path}" "${rbvt_packed_path}"

  log "Done: ${model_name}"
  log "SqueezeLLM: ${sqllm_packed_path}"
  log "LNQ plain: ${lnq_packed_path}"
  log "RBVT-SqueezeLLM: ${rbvt_packed_path}"
  log "Repo eval: ${repo_results_file}"
  log "NonUQuant eval dir: ${nonuq_output_dir}"
}

main() {
  log "Models: ${MODELS}"
  log "Config: bits=${BITS}, RedPajama ${NUM_EXAMPLES}x${SEQ_LEN}, LNQ g=${LNQ_NUM_GROUPS} iter=${LNQ_NUM_ITERATIONS} cd=${LNQ_CD_CYCLES}, repo_eval_ctx=${REPO_EVAL_CONTEXT}"
  log "RedPajama source: ${OFFICIAL_REDPJ_DATASET}${OFFICIAL_REDPJ_CONFIG:+/${OFFICIAL_REDPJ_CONFIG}}, split=${OFFICIAL_REDPJ_SPLIT}, streaming shuffle buffer=${OFFICIAL_REDPJ_SHUFFLE_BUFFER}"
  for model in ${MODELS}; do
    run_model_job "${model}"
  done
  log "All jobs done."
}

main "$@"
