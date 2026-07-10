#!/usr/bin/env bash
set -Eeuo pipefail

JOB_NAME="${JOB_NAME:-llama2-7b-redpajama-bv-sq}"
trap 'echo "[${JOB_NAME}] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
MODEL_LABEL="${MODEL_LABEL:-llama2_7b}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
BIT="${BIT:-3}"
DATASET="${DATASET:-redpajama}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-4096}"
SEED="${SEED:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${MODEL_LABEL}_${BIT}bit_redpajama_bv_sq}"
CACHE_ROOT="${CACHE_ROOT:-cache/${MODEL_LABEL}_${BIT}bit_redpajama_bv_sq}"
CHUNK_DIR="${CHUNK_DIR:-${OUTPUT_ROOT}/chunks}"
SQ_DIR="${SQ_DIR:-${OUTPUT_ROOT}/squeezellm_w${BIT}}"
FISHER_DIR="${FISHER_DIR:-${OUTPUT_ROOT}/fisher_${DATASET}_s${NSAMPLES}_blk${SEQLEN}}"
PPL_DIR="${PPL_DIR:-${OUTPUT_ROOT}/ppl}"
PACK_DIR="${PACK_DIR:-${OUTPUT_ROOT}/packed}"
STATS_PATH="${STATS_PATH:-${OUTPUT_ROOT}/bv_stats_${DATASET}_s${NSAMPLES}_blk${SEQLEN}.pt}"

REDPAJAMA_TOKEN_CACHE="${REDPAJAMA_TOKEN_CACHE:-cache/tokens/Llama-2-7b-hf-redpajama_s1024_blk4096.pt}"
if [[ -z "${CALIB_TOKENS_PATH:-}" && -s "${REDPAJAMA_TOKEN_CACHE}" ]]; then
  export CALIB_TOKENS_PATH="${REDPAJAMA_TOKEN_CACHE}"
fi

DEVICE="${DEVICE:-auto}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-80}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-16000}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
RUN_SQLLM_BASELINE="${RUN_SQLLM_BASELINE:-1}"
FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-1}"
FISHER_LAYERS_PER_PASS="${FISHER_LAYERS_PER_PASS:-1}"
FISHER_ACCUM_DEVICE="${FISHER_ACCUM_DEVICE:-cuda}"
FISHER_EMPTY_CACHE_INTERVAL="${FISHER_EMPTY_CACHE_INTERVAL:-0}"
FISHER_GRADIENT_CHECKPOINTING="${FISHER_GRADIENT_CHECKPOINTING:-on}"
FISHER_MODEL_DTYPE="${FISHER_MODEL_DTYPE:-default}"
BV_MODEL_DTYPE="${BV_MODEL_DTYPE:-float16}"
BV_N_CALIB="${BV_N_CALIB:-1024}"
BV_BATCH_SIZE="${BV_BATCH_SIZE:-1}"
CPU_COUNT="${CPU_COUNT:-16}"
ROW_CHUNKSIZE="${ROW_CHUNKSIZE:-8}"
BV_H_SOURCE="${BV_H_SOURCE:-variance}"
BV_H_FLOOR="${BV_H_FLOOR:-1e-8}"
BV_MIN_SIZE="${BV_MIN_SIZE:-1}"
BV_EPS="${BV_EPS:-1e-12}"

BV_SQ_VARIANTS="${BV_SQ_VARIANTS:-greedy_l1 hier_l1 greedy_l0 greedy_l1_rbvt}"
OVERWRITE_CHUNKS="${OVERWRITE_CHUNKS:-0}"
BV_OVERWRITE="${BV_OVERWRITE:-0}"
BV_OVERWRITE_STATS="${BV_OVERWRITE_STATS:-0}"
RBVT_OVERWRITE="${RBVT_OVERWRITE:-0}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-0}"

RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_BUDGET_P="${RBVT_BUDGET_P:-1.0}"
RBVT_TARGET_RATIO="${RBVT_TARGET_RATIO:-1.0}"
RBVT_MSE_GUARD="${RBVT_MSE_GUARD:-0}"
RBVT_ROW_CHUNK="${RBVT_ROW_CHUNK:-1024}"
RBVT_GAP_FLOOR="${RBVT_GAP_FLOOR:-1e-8}"

PPL_DATASETS="${PPL_DATASETS:-wikitext2 c4}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-1}"
PPL_DENSE_DTYPE="${PPL_DENSE_DTYPE:-float16}"
NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-2048}"
NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"
LIMIT_TOKENS="${LIMIT_TOKENS:-0}"

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${PPL_DIR}" "${PACK_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${JOB_NAME}] $*"
}

select_device() {
  if [[ "${DEVICE}" != "auto" ]]; then
    echo "${DEVICE}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi
  local best
  best="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v min_free="${GPU_MIN_FREE_MB}" -v max_util="${GPU_MAX_UTIL}" '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
          if ($2 >= min_free && $3 <= max_util) {
            if ($2 > best_free) { best_free=$2; best=$1; }
          }
        }
        END { if (best != "") print best; }
      ')"
  if [[ -z "${best}" ]]; then
    best="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 > best_free) {best_free=$2; best=$1}} END {print best}')"
  fi
  echo "cuda:${best}"
}

count_chunks() {
  find "${CHUNK_DIR}" -maxdepth 1 -name 'layer_*.pt' 2>/dev/null | wc -l | tr -d ' '
}

count_files() {
  local dir="$1"
  local pattern="$2"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -name "${pattern}" | wc -l | tr -d ' '
}

stage_complete() {
  local dir="$1"
  local pattern="$2"
  local expected="$3"
  local count
  count="$(count_files "${dir}" "${pattern}")"
  [[ "${count}" -ge "${expected}" ]]
}

variant_folder() {
  case "$1" in
    greedy_l1) echo "${OUTPUT_ROOT}/bv_sq_greedy_w${BIT}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_lambda1.0" ;;
    hier_l1) echo "${OUTPUT_ROOT}/bv_sq_hier_w${BIT}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_lambda1.0" ;;
    greedy_l0) echo "${OUTPUT_ROOT}/bv_sq_greedy_w${BIT}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_lambda0.0" ;;
    greedy_l1_rbvt) echo "${OUTPUT_ROOT}/bv_sq_greedy_w${BIT}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_lambda1.0_rbvt" ;;
    *) echo "Unknown BV_SQ variant: $1" >&2; exit 2 ;;
  esac
}

variant_solver() {
  case "$1" in
    greedy_l1|greedy_l0|greedy_l1_rbvt) echo "greedy" ;;
    hier_l1) echo "hier" ;;
  esac
}

variant_lambda() {
  case "$1" in
    greedy_l0) echo "0.0" ;;
    greedy_l1|hier_l1|greedy_l1_rbvt) echo "1.0" ;;
  esac
}

layer_count() {
  "${PYTHON_BIN}" - "$MODEL" <<'PY'
import sys
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
count = getattr(cfg, "num_hidden_layers", None)
if count is None:
    count = getattr(cfg, "n_layer", None)
if count is None:
    raise SystemExit("Cannot infer model layer count from config.")
print(int(count))
PY
}

run_sqllm_baseline() {
  if [[ "${RUN_SQLLM_BASELINE}" != "1" ]]; then
    log "RUN_SQLLM_BASELINE=${RUN_SQLLM_BASELINE}; skipping sqllm baseline"
    return 0
  fi
  if stage_complete "${SQ_DIR}/lut" 'l*.pkl' "${EXPECTED_LAYERS}"; then
    log "SqueezeLLM baseline LUT complete in ${SQ_DIR}/lut; skipping build"
    return 0
  fi
  if stage_complete "${FISHER_DIR}" 'layer_*.pt' "${EXPECTED_LAYERS}"; then
    log "Fisher chunks complete in ${FISHER_DIR}; skipping Fisher"
  else
    log "Collecting Fisher chunks for sqllm baseline: output=${FISHER_DIR}"
    "${PYTHON_BIN}" quantization/fisher.py \
      --model "${MODEL}" \
      --output_path "${FISHER_DIR}" \
      --dataset "${DATASET}" \
      --nsamples "${NSAMPLES}" \
      --seqlen "${SEQLEN}" \
      --seed "${SEED}" \
      --cache_dir "${CACHE_ROOT}/tokens" \
      --device "${RUN_DEVICE}" \
      --batch_size "${FISHER_BATCH_SIZE}" \
      --layers_per_pass "${FISHER_LAYERS_PER_PASS}" \
      --accum_device "${FISHER_ACCUM_DEVICE}" \
      --empty_cache_interval "${FISHER_EMPTY_CACHE_INTERVAL}" \
      --gradient_checkpointing "${FISHER_GRADIENT_CHECKPOINTING}" \
      --model_dtype "${FISHER_MODEL_DTYPE}" \
      --attn_implementation "${ATTN_IMPLEMENTATION}"
  fi
  log "Building/resuming sqllm baseline LUT: output=${SQ_DIR}"
  "${PYTHON_BIN}" quantization/nuq.py \
    --model_type "${MODEL_TYPE}" \
    --model "${CHUNK_DIR}" \
    --gradient "${FISHER_DIR}" \
    --bit "${BIT}" \
    --output_folder "${SQ_DIR}"
}

run_bv_variant() {
  local variant="$1"
  [[ "${variant}" == "greedy_l1_rbvt" ]] && return 0
  local out_dir solver lambda
  local -a overwrite_args stats_args
  out_dir="$(variant_folder "${variant}")"
  solver="$(variant_solver "${variant}")"
  lambda="$(variant_lambda "${variant}")"
  overwrite_args=()
  stats_args=()
  [[ "${BV_OVERWRITE}" == "1" ]] && overwrite_args+=(--overwrite)
  [[ "${BV_OVERWRITE_STATS}" == "1" ]] && stats_args+=(--overwrite_stats)

  log "Running BV-SQ ${variant}: solver=${solver}, bias_lambda=${lambda}, output=${out_dir}"
  "${PYTHON_BIN}" quantization/bv_sq.py all \
    --model "${MODEL}" \
    --model_chunks "${CHUNK_DIR}" \
    --output_folder "${out_dir}" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --nsamples "${NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --seed "${SEED}" \
    --cache_dir "${CACHE_ROOT}/tokens" \
    --stats_path "${STATS_PATH}" \
    --device "${RUN_DEVICE}" \
    --model_dtype "${BV_MODEL_DTYPE}" \
    --n_calib "${BV_N_CALIB}" \
    --batch_size "${BV_BATCH_SIZE}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --bit "${BIT}" \
    --solver "${solver}" \
    --bias_lambda "${lambda}" \
    --min_size "${BV_MIN_SIZE}" \
    --eps "${BV_EPS}" \
    --h_source "${BV_H_SOURCE}" \
    --h_floor "${BV_H_FLOOR}" \
    --cpu_count "${CPU_COUNT}" \
    --row_chunksize "${ROW_CHUNKSIZE}" \
    "${overwrite_args[@]}" \
    "${stats_args[@]}"
}

run_rbvt_variant() {
  local input_dir output_dir
  local -a overwrite_args guard_args
  input_dir="$(variant_folder greedy_l1)"
  output_dir="$(variant_folder greedy_l1_rbvt)"
  overwrite_args=()
  guard_args=()
  [[ "${RBVT_OVERWRITE}" == "1" ]] && overwrite_args+=(--overwrite)
  [[ "${BV_OVERWRITE_STATS}" == "1" ]] && overwrite_args+=(--overwrite_stats)
  [[ "${RBVT_MSE_GUARD}" == "1" ]] && guard_args+=(--rbvt_mse_guard)

  log "Running BV-SQ + RBVT final assignment: input=${input_dir}, output=${output_dir}"
  "${PYTHON_BIN}" quantization/rbvt_squeezellm.py all \
    --model "${MODEL}" \
    --model_chunks "${CHUNK_DIR}" \
    --input_lut "${input_dir}" \
    --output_folder "${output_dir}" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --nsamples "${NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --seed "${SEED}" \
    --cache_dir "${CACHE_ROOT}/tokens" \
    --stats_path "${STATS_PATH}" \
    --device "${RUN_DEVICE}" \
    --model_dtype "${BV_MODEL_DTYPE}" \
    --n_calib "${BV_N_CALIB}" \
    --batch_size "${BV_BATCH_SIZE}" \
    --rbvt_lambda "${RBVT_LAMBDA}" \
    --rbvt_budget_p "${RBVT_BUDGET_P}" \
    --rbvt_target_ratio "${RBVT_TARGET_RATIO}" \
    --row_chunk "${RBVT_ROW_CHUNK}" \
    --gap_floor "${RBVT_GAP_FLOOR}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    "${guard_args[@]}" \
    "${overwrite_args[@]}"
}

eval_sqllm_baseline() {
  if [[ "${RUN_SQLLM_BASELINE}" != "1" ]]; then
    return 0
  fi
  local out_file checkpoint
  out_file="${PPL_DIR}/sqllm_nonuquantfix_dense_lut_ppl.json"
  checkpoint="${PACK_DIR}/${MODEL_LABEL}_sqllm_w${BIT}.pt"
  if [[ -s "${out_file}" && "${EVAL_OVERWRITE}" != "1" ]]; then
    log "Skipping eval for sqllm; existing ${out_file}"
    return 0
  fi
  if [[ ! -d "${SQ_DIR}/lut" ]]; then
    log "Skipping eval for sqllm; missing ${SQ_DIR}/lut"
    return 0
  fi
  log "Evaluating sqllm baseline with nonuquantfix dense_lut PPL on ${RUN_DEVICE}: ${PPL_DATASETS}"
  "${PYTHON_BIN}" quantization/eval_nonuquantfix_ppl.py \
    --model "${MODEL}" \
    --checkpoint "${checkpoint}" \
    --wbits "${BIT}" \
    --model_type "${MODEL_TYPE}" \
    --backend dense_lut \
    --lut_folder "${SQ_DIR}" \
    --dense_dtype "${PPL_DENSE_DTYPE}" \
    --datasets ${PPL_DATASETS} \
    --device "${RUN_DEVICE}" \
    --stride "${NONUQ_STRIDE}" \
    --max_length "${NONUQ_MAX_LENGTH}" \
    --batch_size "${PPL_BATCH_SIZE}" \
    --c4_samples "${NONUQ_C4_SAMPLES}" \
    --limit_tokens "${LIMIT_TOKENS}" \
    --output_file "${out_file}"
}

eval_variant() {
  local variant="$1"
  local lut_dir out_file checkpoint
  lut_dir="$(variant_folder "${variant}")"
  out_file="${PPL_DIR}/${variant}_nonuquantfix_dense_lut_ppl.json"
  checkpoint="${PACK_DIR}/${MODEL_LABEL}_${variant}_w${BIT}.pt"
  if [[ -s "${out_file}" && "${EVAL_OVERWRITE}" != "1" ]]; then
    log "Skipping eval for ${variant}; existing ${out_file}"
    return 0
  fi
  if [[ ! -d "${lut_dir}/lut" ]]; then
    log "Skipping eval for ${variant}; missing ${lut_dir}/lut"
    return 0
  fi
  log "Evaluating ${variant} with nonuquantfix dense_lut PPL on ${RUN_DEVICE}: ${PPL_DATASETS}"
  "${PYTHON_BIN}" quantization/eval_nonuquantfix_ppl.py \
    --model "${MODEL}" \
    --checkpoint "${checkpoint}" \
    --wbits "${BIT}" \
    --model_type "${MODEL_TYPE}" \
    --backend dense_lut \
    --lut_folder "${lut_dir}" \
    --dense_dtype "${PPL_DENSE_DTYPE}" \
    --datasets ${PPL_DATASETS} \
    --device "${RUN_DEVICE}" \
    --stride "${NONUQ_STRIDE}" \
    --max_length "${NONUQ_MAX_LENGTH}" \
    --batch_size "${PPL_BATCH_SIZE}" \
    --c4_samples "${NONUQ_C4_SAMPLES}" \
    --limit_tokens "${LIMIT_TOKENS}" \
    --output_file "${out_file}"
}

RUN_DEVICE="$(select_device)"
log "model=${MODEL}; calib=${DATASET} n=${NSAMPLES} seqlen=${SEQLEN}; device=${RUN_DEVICE}"
if [[ "${DATASET}" == "redpajama" && -z "${CALIB_TOKENS_PATH:-}" ]]; then
  echo "Missing RedPajama token cache. Set CALIB_TOKENS_PATH or REDPAJAMA_TOKEN_CACHE." >&2
  exit 1
fi
if [[ -n "${CALIB_TOKENS_PATH:-}" ]]; then
  log "Using calibration token cache: ${CALIB_TOKENS_PATH}"
fi

EXPECTED_LAYERS="$(layer_count)"
if [[ "${OVERWRITE_CHUNKS}" == "1" || "$(count_chunks)" != "${EXPECTED_LAYERS}" ]]; then
  log "Chunking model into ${CHUNK_DIR}"
  declare -a chunk_args=()
  [[ "${OVERWRITE_CHUNKS}" == "1" ]] && chunk_args+=(--overwrite)
  "${PYTHON_BIN}" quantization/chunk_models.py \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --output_path "${CHUNK_DIR}" \
    "${chunk_args[@]}"
else
  log "Reusing existing chunks in ${CHUNK_DIR} ($(count_chunks)/${EXPECTED_LAYERS})"
fi

run_sqllm_baseline

for variant in ${BV_SQ_VARIANTS}; do
  run_bv_variant "${variant}"
done

for variant in ${BV_SQ_VARIANTS}; do
  if [[ "${variant}" == "greedy_l1_rbvt" ]]; then
    run_bv_variant greedy_l1
    run_rbvt_variant
  fi
done

eval_sqllm_baseline

for variant in ${BV_SQ_VARIANTS}; do
  eval_variant "${variant}"
done

log "Done. PPL JSON files are in ${PPL_DIR}"
