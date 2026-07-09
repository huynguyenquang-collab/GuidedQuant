#!/usr/bin/env bash
set -Eeuo pipefail

JOB_NAME="${JOB_NAME:-llama2-7b-llama3-8b-c4-sqllm-lnq-rbvt}"
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
MODEL_TARGETS="${MODEL_TARGETS:-llama2_7b llama3_8b}"
LLAMA2_MODEL="${LLAMA2_MODEL:-meta-llama/Llama-2-7b-hf}"
LLAMA3_MODEL="${LLAMA3_MODEL:-meta-llama/Meta-Llama-3-8B}"
BIT="${BIT:-3}"
DATASET="${DATASET:-c4}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"

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
RBVT_N_CALIB="${RBVT_N_CALIB:-128}"
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
  local cache_env=()

  if [[ -s "${token_cache}" ]]; then
    cache_env=(
      "LEGACY_CALIB_TOKENS=${token_cache}"
      "CALIB_TOKENS_PATH=${token_cache}"
    )
    log "Running ${label}: model=${model}; using C4 token cache=${token_cache}; targets=${PPL_TARGETS}"
  else
    log "Running ${label}: model=${model}; no C4 token cache at ${token_cache}; will build/cache C4 tokens once"
  fi

  env \
    "${cache_env[@]}" \
    MODEL="${model}" \
    MODEL_LABEL="${label}" \
    MODEL_TYPE="llama" \
    BIT="${BIT}" \
    DATASET="${DATASET}" \
    NSAMPLES="${NSAMPLES}" \
    SEQLEN="${SEQLEN}" \
    JOB_NAME="${label}-c4-sqllm-lnq-rbvt" \
    OUTPUT_ROOT="outputs/${label}_3bit_c4_original_sqllm_lnq_rbvt" \
    CACHE_ROOT="cache/${label}_3bit_c4_original_sqllm_lnq_rbvt" \
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

for target in ${MODEL_TARGETS}; do
  case "${target}" in
    llama2_7b)
      run_model "${LLAMA2_MODEL}" "${target}" \
        "cache/tokens/Llama-2-7b-hf-c4_s128_blk2048.pt"
      ;;
    llama3_8b)
      run_model "${LLAMA3_MODEL}" "${target}" \
        "cache/tokens/Meta-Llama-3-8B-c4_s128_blk2048.pt"
      ;;
    *)
      echo "Unknown MODEL_TARGETS entry: ${target}" >&2
      exit 2
      ;;
  esac
done

log "Done. Results:"
log "  outputs/llama2_7b_3bit_c4_original_sqllm_lnq_rbvt/ppl"
log "  outputs/llama3_8b_3bit_c4_original_sqllm_lnq_rbvt/ppl"
