#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[llama-c4-guidedquant] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_DIR="${CACHE_DIR:-cache}"

BITS="${BITS:-3}"
DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"

LNQ_NUM_GROUPS="${LNQ_NUM_GROUPS:-1}"
LNQ_NUM_ITERATIONS="${LNQ_NUM_ITERATIONS:-3}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"

RBVT_N_CALIB="${RBVT_N_CALIB:-128}"
RBVT_BATCH_SIZE="${RBVT_BATCH_SIZE:-4}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
RBVT_ROW_CHUNK="${RBVT_ROW_CHUNK:-4096}"
RBVT_GAP_FLOOR="${RBVT_GAP_FLOOR:-1e-8}"
RBVT_OVERWRITE_STATS="${RBVT_OVERWRITE_STATS:-0}"

NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-2048}"
NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"
NONUQ_DTYPE="${NONUQ_DTYPE:-float16}"
NONUQ_DEVICE="${NONUQ_DEVICE:-cuda}"

RUN_SQLLM="${RUN_SQLLM:-1}"
RUN_LNQ="${RUN_LNQ:-1}"
RUN_RBVT="${RUN_RBVT:-1}"
RUN_NONUQ_EVAL="${RUN_NONUQ_EVAL:-1}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-0}"
OVERWRITE_QUANT="${OVERWRITE_QUANT:-0}"
OVERWRITE_PACK="${OVERWRITE_PACK:-0}"

MODEL_SPECS="${MODEL_SPECS:-llama2_7b=meta-llama/Llama-2-7b-hf;llama3_8b=meta-llama/Meta-Llama-3-8B}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

log() {
  local job="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${job}] $*"
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

maybe_overwrite_args() {
  local -n out_ref="$1"
  out_ref=()
  if [[ "${OVERWRITE_QUANT}" == "1" ]]; then
    out_ref+=(--overwrite_quantize)
  fi
  if [[ "${OVERWRITE_PACK}" == "1" ]]; then
    out_ref+=(--overwrite_pack)
  fi
}

run_one_model() {
  local label="$1"
  local model_name="$2"
  local job="${label}-3bit-c4-guidedquant"
  local model_basename="${model_name##*/}"

  local token_path="${CACHE_DIR}/tokens/${model_basename}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"
  local sqllm_quantized_path="${CACHE_DIR}/quantized/${model_basename}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
  local sqllm_packed_path="${CACHE_DIR}/packed/anyprec-${model_basename}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"

  local lnq_quantized_path="${CACHE_DIR}/layerwise_quantized/${model_basename}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"
  local lnq_packed_path="${CACHE_DIR}/layerwise_packed/layerwise-${model_basename}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"

  local rbvt_quantized_path="${CACHE_DIR}/rbvt_sqllm_quantized/${model_basename}-w${BITS}-rbvt-sqllm-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}"
  local rbvt_packed_path="${CACHE_DIR}/rbvt_sqllm_packed/anyprec-rbvt-sqllm-${model_basename}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}"
  local rbvt_stats_path="${CACHE_DIR}/rbvt_sqllm_stats/${model_basename}-w${BITS}-rbvt-sqllm-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA}_n${RBVT_N_CALIB}.pt"

  local nonuq_output_dir="outputs/${label}_sqllm_lnq_rbvt_c4_s${NUM_EXAMPLES}_blk${SEQ_LEN}_${BITS}bit_nonuq"

  log "${job}" "model=${model_name}; calib=${DATASET} n=${NUM_EXAMPLES} seqlen=${SEQ_LEN}; eval=nonuquantfix"

  if [[ "${CLEAN_OUTPUTS}" == "1" ]]; then
    log "${job}" "Cleaning generated artifacts for this job"
    rm -rf \
      "${sqllm_quantized_path}" \
      "${sqllm_packed_path}" \
      "${lnq_quantized_path}" \
      "${lnq_packed_path}" \
      "${rbvt_quantized_path}" \
      "${rbvt_packed_path}" \
      "${rbvt_stats_path}" \
      "${nonuq_output_dir}"
  fi

  local overwrite_args=()
  maybe_overwrite_args overwrite_args

  if [[ "${RUN_SQLLM}" == "1" ]]; then
    if [[ -d "${sqllm_packed_path}" && "${OVERWRITE_PACK}" != "1" ]]; then
      log "${job}" "Reusing SqueezeLLM packed model: ${sqllm_packed_path}"
    else
      log "${job}" "Running SqueezeLLM, ${BITS}-bit"
      "${PYTHON_BIN}" quantize.py "${model_name}" \
        --seed_precision "${BITS}" \
        --parent_precision "${BITS}" \
        --cache_dir "${CACHE_DIR}" \
        --dataset "${DATASET}" \
        --seq_len "${SEQ_LEN}" \
        --num_examples "${NUM_EXAMPLES}" \
        "${overwrite_args[@]}"
      cleanup_cuda
    fi
  else
    log "${job}" "RUN_SQLLM=0; using existing SqueezeLLM artifacts"
  fi

  if [[ "${RUN_LNQ}" == "1" ]]; then
    if [[ -d "${lnq_packed_path}" && "${OVERWRITE_PACK}" != "1" ]]; then
      log "${job}" "Reusing LNQ packed model: ${lnq_packed_path}"
    else
      log "${job}" "Running plain LNQ on top of SqueezeLLM init"
      "${PYTHON_BIN}" layerwise_nuq.py "${model_name}" \
        --seed_precision "${BITS}" \
        --cache_dir "${CACHE_DIR}" \
        --dataset "${DATASET}" \
        --seq_len "${SEQ_LEN}" \
        --num_examples "${NUM_EXAMPLES}" \
        --num_groups "${LNQ_NUM_GROUPS}" \
        --num_iterations "${LNQ_NUM_ITERATIONS}" \
        --cd_cycles "${LNQ_CD_CYCLES}" \
        --is_nosal true \
        "${overwrite_args[@]}"
      cleanup_cuda
    fi
  else
    log "${job}" "RUN_LNQ=0; using existing LNQ artifact"
  fi

  if [[ "${RUN_RBVT}" == "1" ]]; then
    if [[ ! -s "${token_path}" ]]; then
      echo "Missing calibration tokens for RBVT: ${token_path}. Run SqueezeLLM first or set RUN_SQLLM=1." >&2
      exit 1
    fi
    if [[ -d "${rbvt_packed_path}" && "${OVERWRITE_PACK}" != "1" ]]; then
      log "${job}" "Reusing RBVT-SqueezeLLM packed model: ${rbvt_packed_path}"
    else
      local rbvt_stats_args=()
      if [[ "${RBVT_OVERWRITE_STATS}" == "1" ]]; then
        rbvt_stats_args+=(--overwrite-stats)
      fi
      log "${job}" "Running RBVT-SqueezeLLM on top of SqueezeLLM"
      "${PYTHON_BIN}" rbvt_squeezellm.py \
        --model "${model_name}" \
        --bits "${BITS}" \
        --cache-dir "${CACHE_DIR}" \
        --dataset "${DATASET}" \
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
        "${rbvt_stats_args[@]}"
      cleanup_cuda
    fi
  else
    log "${job}" "RUN_RBVT=0; using existing RBVT artifact"
  fi

  if [[ "${RUN_NONUQ_EVAL}" == "1" ]]; then
    mkdir -p "${nonuq_output_dir}"
    log "${job}" "Evaluating SqueezeLLM with NonUQuantFix-style PPL"
    "${PYTHON_BIN}" scripts/eval_nonuquant_style_ppl.py \
      --model-path "${sqllm_packed_path}" \
      --model-name "squeezellm_${model_basename}" \
      --tokenizer-path "${model_name}" \
      --datasets wikitext2 c4 \
      --precision "${BITS}" \
      --dtype "${NONUQ_DTYPE}" \
      --device "${NONUQ_DEVICE}" \
      --max-length "${NONUQ_MAX_LENGTH}" \
      --stride "${NONUQ_STRIDE}" \
      --c4-samples "${NONUQ_C4_SAMPLES}" \
      --output-file "${nonuq_output_dir}/squeezellm.json"
    cleanup_cuda

    log "${job}" "Evaluating LNQ plain with NonUQuantFix-style PPL"
    "${PYTHON_BIN}" scripts/eval_nonuquant_style_ppl.py \
      --model-path "${lnq_packed_path}" \
      --model-name "lnq_plain_${model_basename}" \
      --tokenizer-path "${model_name}" \
      --datasets wikitext2 c4 \
      --precision "${BITS}" \
      --dtype "${NONUQ_DTYPE}" \
      --device "${NONUQ_DEVICE}" \
      --max-length "${NONUQ_MAX_LENGTH}" \
      --stride "${NONUQ_STRIDE}" \
      --c4-samples "${NONUQ_C4_SAMPLES}" \
      --output-file "${nonuq_output_dir}/lnq_plain.json"
    cleanup_cuda

    log "${job}" "Evaluating RBVT-SqueezeLLM with NonUQuantFix-style PPL"
    "${PYTHON_BIN}" scripts/eval_nonuquant_style_ppl.py \
      --model-path "${rbvt_packed_path}" \
      --model-name "rbvt_sqllm_${model_basename}" \
      --tokenizer-path "${model_name}" \
      --datasets wikitext2 c4 \
      --precision "${BITS}" \
      --dtype "${NONUQ_DTYPE}" \
      --device "${NONUQ_DEVICE}" \
      --max-length "${NONUQ_MAX_LENGTH}" \
      --stride "${NONUQ_STRIDE}" \
      --c4-samples "${NONUQ_C4_SAMPLES}" \
      --output-file "${nonuq_output_dir}/rbvt_squeezellm.json"
    cleanup_cuda
  else
    log "${job}" "RUN_NONUQ_EVAL=0; skipping PPL eval"
  fi

  log "${job}" "Done"
  log "${job}" "SqueezeLLM: ${sqllm_packed_path}"
  log "${job}" "LNQ plain: ${lnq_packed_path}"
  log "${job}" "RBVT-SqueezeLLM: ${rbvt_packed_path}"
  log "${job}" "NonUQuantFix PPL: ${nonuq_output_dir}"
}

IFS=';' read -r -a specs <<< "${MODEL_SPECS}"
for spec in "${specs[@]}"; do
  [[ -n "${spec}" ]] || continue
  label="${spec%%=*}"
  model="${spec#*=}"
  if [[ -z "${label}" || -z "${model}" || "${label}" == "${model}" ]]; then
    echo "Bad MODEL_SPECS entry: ${spec}; expected label=hf/model" >&2
    exit 1
  fi
  run_one_model "${label}" "${model}"
done
