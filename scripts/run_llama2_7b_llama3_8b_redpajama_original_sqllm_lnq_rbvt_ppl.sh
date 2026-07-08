#!/usr/bin/env bash
set -Eeuo pipefail

JOB_NAME="${JOB_NAME:-llama2-7b-llama3-8b-redpajama-sqllm-lnq-rbvt}"
trap 'echo "[${JOB_NAME}] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SCRIPT="${SCRIPT_DIR}/run_llama2_13b_redpajama_original_sqllm_rbvt_ppl.sh"
if [[ ! -x "${BASE_SCRIPT}" ]]; then
  echo "Missing executable base script: ${BASE_SCRIPT}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/tp/miniconda/envs/nuquant/bin/python}"
BIT="${BIT:-3}"
DATASET="${DATASET:-redpajama}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-4096}"

GPU_DEVICES="${GPU_DEVICES:-auto}"
GPU_MAX_DEVICES="${GPU_MAX_DEVICES:-2}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-90}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-22000}"
HESSIAN_MIN_FREE_MB="${HESSIAN_MIN_FREE_MB:-22000}"
FISHER_LAYERS_PER_PASS="${FISHER_LAYERS_PER_PASS:-4}"
FISHER_GRADIENT_CHECKPOINTING="${FISHER_GRADIENT_CHECKPOINTING:-on}"
FISHER_ACCUM_DEVICE="${FISHER_ACCUM_DEVICE:-cuda}"
FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"

HESSIAN_CALIB_BATCH_SIZE="${HESSIAN_CALIB_BATCH_SIZE:-4}"
HESSIAN_ACTIVATION_STORAGE="${HESSIAN_ACTIVATION_STORAGE:-ram}"
HESSIAN_ACCUM_DEVICE="${HESSIAN_ACCUM_DEVICE:-cuda}"
LNQ_ITERATIONS="${LNQ_ITERATIONS:-3}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"
LNQ_ROW_BLOCK="${LNQ_ROW_BLOCK:-64}"
CPU_COUNT="${CPU_COUNT:-16}"

RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_BUDGET_P="${RBVT_BUDGET_P:-1.0}"
RBVT_TARGET_RATIO="${RBVT_TARGET_RATIO:-1.0}"
RBVT_MSE_GUARD="${RBVT_MSE_GUARD:-0}"
RBVT_N_CALIB="${RBVT_N_CALIB:-1024}"
RBVT_BATCH_SIZE="${RBVT_BATCH_SIZE:-1}"

PPL_TARGETS="${PPL_TARGETS:-squeezellm lnq_plain rbvt_squeeze}"
PPL_BACKEND="${PPL_BACKEND:-dense_lut}"
PPL_EVAL_STYLE="${PPL_EVAL_STYLE:-nonuquantfix}"
PPL_DATASETS="${PPL_DATASETS:-wikitext2 c4}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-1}"
EVAL_PARALLEL_DATASETS="${EVAL_PARALLEL_DATASETS:-1}"
NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-2048}"
NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"

mkdir -p outputs cache

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${JOB_NAME}] $*"
}

run_model() {
  local model="$1"
  local label="$2"
  local token_cache="$3"

  if [[ ! -s "${token_cache}" ]]; then
    echo "Missing RedPajama token cache for ${label}: ${token_cache}" >&2
    exit 1
  fi

  log "Running ${label}: model=${model}; cache=${token_cache}; targets=${PPL_TARGETS}"

  MODEL="${model}" \
  MODEL_LABEL="${label}" \
  MODEL_TYPE="llama" \
  BIT="${BIT}" \
  DATASET="${DATASET}" \
  NSAMPLES="${NSAMPLES}" \
  SEQLEN="${SEQLEN}" \
  JOB_NAME="${label}-redpajama-sqllm-lnq-rbvt" \
  OUTPUT_ROOT="outputs/${label}_3bit_redpajama_original_sqllm_lnq_rbvt" \
  CACHE_ROOT="cache/${label}_3bit_redpajama_original_sqllm_lnq_rbvt" \
  LEGACY_CALIB_TOKENS="${token_cache}" \
  CALIB_TOKENS_PATH="${token_cache}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  GPU_DEVICES="${GPU_DEVICES}" \
  GPU_MAX_DEVICES="${GPU_MAX_DEVICES}" \
  GPU_MAX_UTIL="${GPU_MAX_UTIL}" \
  GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB}" \
  HESSIAN_MIN_FREE_MB="${HESSIAN_MIN_FREE_MB}" \
  FISHER_LAYERS_PER_PASS="${FISHER_LAYERS_PER_PASS}" \
  FISHER_GRADIENT_CHECKPOINTING="${FISHER_GRADIENT_CHECKPOINTING}" \
  FISHER_ACCUM_DEVICE="${FISHER_ACCUM_DEVICE}" \
  FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE}" \
  ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
  HESSIAN_CALIB_BATCH_SIZE="${HESSIAN_CALIB_BATCH_SIZE}" \
  HESSIAN_ACTIVATION_STORAGE="${HESSIAN_ACTIVATION_STORAGE}" \
  HESSIAN_ACCUM_DEVICE="${HESSIAN_ACCUM_DEVICE}" \
  LNQ_ITERATIONS="${LNQ_ITERATIONS}" \
  LNQ_CD_CYCLES="${LNQ_CD_CYCLES}" \
  LNQ_ROW_BLOCK="${LNQ_ROW_BLOCK}" \
  CPU_COUNT="${CPU_COUNT}" \
  RBVT_LAMBDA="${RBVT_LAMBDA}" \
  RBVT_BUDGET_P="${RBVT_BUDGET_P}" \
  RBVT_TARGET_RATIO="${RBVT_TARGET_RATIO}" \
  RBVT_MSE_GUARD="${RBVT_MSE_GUARD}" \
  RBVT_N_CALIB="${RBVT_N_CALIB}" \
  RBVT_BATCH_SIZE="${RBVT_BATCH_SIZE}" \
  PPL_TARGETS="${PPL_TARGETS}" \
  PPL_BACKEND="${PPL_BACKEND}" \
  PPL_EVAL_STYLE="${PPL_EVAL_STYLE}" \
  PPL_DATASETS="${PPL_DATASETS}" \
  PPL_BATCH_SIZE="${PPL_BATCH_SIZE}" \
  EVAL_PARALLEL_DATASETS="${EVAL_PARALLEL_DATASETS}" \
  NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH}" \
  NONUQ_STRIDE="${NONUQ_STRIDE}" \
  NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES}" \
  bash "${BASE_SCRIPT}"
}

run_model \
  "meta-llama/Llama-2-7b-hf" \
  "llama2_7b" \
  "cache/tokens/Llama-2-7b-hf-redpajama_s1024_blk4096.pt"

run_model \
  "meta-llama/Meta-Llama-3-8B" \
  "llama3_8b" \
  "cache/tokens/Meta-Llama-3-8B-redpajama_s1024_blk4096.pt"

log "Done. Results:"
log "  outputs/llama2_7b_3bit_redpajama_original_sqllm_lnq_rbvt/ppl"
log "  outputs/llama3_8b_3bit_redpajama_original_sqllm_lnq_rbvt/ppl"
