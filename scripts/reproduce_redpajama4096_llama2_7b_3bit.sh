#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[repro] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Paper setup for Table 3 weight-only scalar PTQ:
# - Llama-2-7B
# - RedPajama calibration, 1024 samples x 4096 tokens
# - 3-bit non-uniform scalar quantization
# - SqueezeLLM baseline and plain LNQ, not LNQ + GuidedQuant
# - Perplexity evaluation on WikiText2 and C4 with context length 4096
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
MODEL_BASENAME="${MODEL_NAME##*/}"
BITS="${BITS:-3}"
LNQ_NUM_GROUPS="${LNQ_NUM_GROUPS:-1}"
LNQ_NUM_ITERATIONS="${LNQ_NUM_ITERATIONS:-2}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
CACHE_DIR="${CACHE_DIR:-cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ "${PYTHON_BIN}" == "python" ]] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
fi

RESULTS_FILE="${RESULTS_FILE:-results_llama2_7b_redpajama4096_3bit_ctx4096.json}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-${CACHE_DIR}/repro_eval_llama2_7b_redpajama4096_3bit}"

# Set to 1 on a fresh server if dependencies have not been installed yet.
INSTALL_DEPS="${INSTALL_DEPS:-0}"
INSTALL_AP_GEMV="${INSTALL_AP_GEMV:-0}"

# Keep this as 1 for a clean reproduction; it only removes paths for this exact run.
REPRO_CLEAN="${REPRO_CLEAN:-1}"
RUN_QUANTIZATION="${RUN_QUANTIZATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
REDO_EVAL="${REDO_EVAL:-1}"
DOWNSTREAM="${DOWNSTREAM:-0}"
SKIP_ENV_CHECK="${SKIP_ENV_CHECK:-0}"
ALLOW_NON_PAPER_CONFIG="${ALLOW_NON_PAPER_CONFIG:-0}"
MEMORY_CLEANUP="${MEMORY_CLEANUP:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TOKEN_URL="https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Llama-2-7b-hf-redpajama_s1024_blk4096.pt"
TOKEN_PATH="${CACHE_DIR}/tokens/${MODEL_BASENAME}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"

GRADIENTS_PATH="${CACHE_DIR}/gradients/${MODEL_BASENAME}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"
HESSIANS_PATH="${CACHE_DIR}/hessians/${MODEL_BASENAME}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_nosal"
SQLLM_QUANTIZED_PATH="${CACHE_DIR}/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
SQLLM_PACKED_PATH="${CACHE_DIR}/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
LNQ_QUANTIZED_PATH="${CACHE_DIR}/layerwise_quantized/${MODEL_BASENAME}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"
LNQ_PACKED_PATH="${CACHE_DIR}/layerwise_packed/layerwise-${MODEL_BASENAME}-w${BITS}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${LNQ_NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [repro] $*"
}

cleanup_after_stage() {
  local stage_name="$1"

  if [[ "${MEMORY_CLEANUP}" != "1" ]]; then
    return
  fi

  log "Cleaning up after ${stage_name}"
  sync || true
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

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits || true
  fi
}

download_file() {
  local url="$1"
  local output="$2"

  mkdir -p "$(dirname "${output}")"
  if [[ -s "${output}" ]]; then
    log "Using existing calibration file: ${output}"
    return
  fi

  log "Downloading calibration file: ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output}"
  else
    echo "Neither wget nor curl is installed; cannot download calibration data." >&2
    exit 1
  fi
}

maybe_install_deps() {
  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    log "Installing Python requirements from requirements.txt"
    "${PYTHON_BIN}" -m pip install -r requirements.txt
  fi

  if [[ "${INSTALL_AP_GEMV}" == "1" ]]; then
    log "Installing prebuilt ap_gemv kernel for CUDA 12.4"
    "${PYTHON_BIN}" -m pip install ap-gemv -i https://jinukkim.me/whl/cu124
  fi
}

assert_paper_config() {
  if [[ "${ALLOW_NON_PAPER_CONFIG}" == "1" ]]; then
    log "ALLOW_NON_PAPER_CONFIG=1; not enforcing paper defaults"
    return
  fi

  if [[ "${MODEL_BASENAME}" != "Llama-2-7b-hf" || "${BITS}" != "3" || "${LNQ_NUM_GROUPS}" != "1" || "${LNQ_NUM_ITERATIONS}" != "2" || "${LNQ_CD_CYCLES}" != "4" || "${SEQ_LEN}" != "4096" || "${NUM_EXAMPLES}" != "1024" ]]; then
    cat >&2 <<EOF
This script is locked to the paper setup:
  MODEL_BASENAME=Llama-2-7b-hf
  BITS=3
  LNQ_NUM_GROUPS=1
  LNQ_NUM_ITERATIONS=2
  LNQ_CD_CYCLES=4
  SEQ_LEN=4096
  NUM_EXAMPLES=1024

Current:
  MODEL_NAME=${MODEL_NAME}
  MODEL_BASENAME=${MODEL_BASENAME}
  BITS=${BITS}
  LNQ_NUM_GROUPS=${LNQ_NUM_GROUPS}
  LNQ_NUM_ITERATIONS=${LNQ_NUM_ITERATIONS}
  LNQ_CD_CYCLES=${LNQ_CD_CYCLES}
  SEQ_LEN=${SEQ_LEN}
  NUM_EXAMPLES=${NUM_EXAMPLES}

Set ALLOW_NON_PAPER_CONFIG=1 only if you intentionally want a non-paper run.
EOF
    exit 1
  fi
}

check_environment() {
  if [[ "${SKIP_ENV_CHECK}" == "1" ]]; then
    log "Skipping environment checks"
    return
  fi

  log "Checking Python/CUDA/package environment"
  "${PYTHON_BIN}" - <<'PY'
import importlib
import sys

missing = []
for package in [
    "torch",
    "transformers",
    "datasets",
    "lm_eval",
    "flash1dkmeans",
    "any_precision",
]:
    try:
        importlib.import_module(package)
    except Exception as exc:
        missing.append(f"{package}: {exc}")

try:
    importlib.import_module("ap_gemv")
except Exception as exc:
    missing.append(f"ap_gemv: {exc}")

if missing:
    print("Missing or broken packages:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print("Run with INSTALL_DEPS=1 INSTALL_AP_GEMV=1, or install the repo requirements and ap_gemv manually.", file=sys.stderr)
    sys.exit(1)

import torch
import transformers
import datasets
import lm_eval

if not torch.cuda.is_available():
    print("CUDA is not available. Quantization/eval for Llama-2-7B is expected to run on CUDA GPUs.", file=sys.stderr)
    sys.exit(1)

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, devices={torch.cuda.device_count()}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__}")
print(f"lm_eval={getattr(lm_eval, '__version__', 'unknown')}")
PY
}

clean_repro_outputs() {
  if [[ "${REPRO_CLEAN}" != "1" ]]; then
    log "REPRO_CLEAN=0; keeping existing generated artifacts"
    return
  fi

  log "Removing generated artifacts for this exact reproduction"
  rm -rf \
    "${GRADIENTS_PATH}" \
    "${HESSIANS_PATH}" \
    "${SQLLM_QUANTIZED_PATH}" \
    "${SQLLM_PACKED_PATH}" \
    "${LNQ_QUANTIZED_PATH}" \
    "${LNQ_PACKED_PATH}" \
    "${EVAL_CACHE_DIR}" \
    "${RESULTS_FILE}"
}

verify_calibration() {
  log "Verifying calibration tensor can be loaded: ${TOKEN_PATH}"
  "${PYTHON_BIN}" - "${TOKEN_PATH}" <<'PY'
import sys
import torch

path = sys.argv[1]
tokens = torch.load(path, map_location="cpu")
shape = tuple(tokens.shape) if hasattr(tokens, "shape") else None
print(f"calibration_type={type(tokens).__name__}, shape={shape}")
PY
}

run_squeezellm() {
  log "Running SqueezeLLM: model=${MODEL_NAME}, bits=${BITS}, RedPajama ${NUM_EXAMPLES}x${SEQ_LEN}"
  "${PYTHON_BIN}" quantize.py "${MODEL_NAME}" \
    --seed_precision "${BITS}" \
    --parent_precision "${BITS}" \
    --cache_dir "${CACHE_DIR}" \
    --dataset redpajama \
    --seq_len "${SEQ_LEN}" \
    --num_examples "${NUM_EXAMPLES}"

  [[ -d "${SQLLM_PACKED_PATH}" ]] || {
    echo "Expected SqueezeLLM packed model was not created: ${SQLLM_PACKED_PATH}" >&2
    exit 1
  }
  cleanup_after_stage "SqueezeLLM"
}

run_lnq_plain() {
  [[ -d "${SQLLM_QUANTIZED_PATH}" ]] || {
    echo "LNQ initialization cache is missing. SqueezeLLM must create: ${SQLLM_QUANTIZED_PATH}" >&2
    exit 1
  }

  log "Running plain LNQ: model=${MODEL_NAME}, bits=${BITS}, RedPajama ${NUM_EXAMPLES}x${SEQ_LEN}"
  "${PYTHON_BIN}" layerwise_nuq.py "${MODEL_NAME}" \
    --seed_precision "${BITS}" \
    --cache_dir "${CACHE_DIR}" \
    --dataset redpajama \
    --seq_len "${SEQ_LEN}" \
    --num_examples "${NUM_EXAMPLES}" \
    --num_groups "${LNQ_NUM_GROUPS}" \
    --num_iterations "${LNQ_NUM_ITERATIONS}" \
    --cd_cycles "${LNQ_CD_CYCLES}" \
    --is_nosal true

  [[ -d "${LNQ_PACKED_PATH}" ]] || {
    echo "Expected LNQ packed model was not created: ${LNQ_PACKED_PATH}" >&2
    exit 1
  }
  cleanup_after_stage "LNQ"
}

absolute_path() {
  "${PYTHON_BIN}" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

prepare_eval_cache() {
  log "Preparing isolated eval cache: ${EVAL_CACHE_DIR}"
  mkdir -p "${EVAL_CACHE_DIR}/packed" "${EVAL_CACHE_DIR}/layerwise_packed"
  ln -sfn "$(absolute_path "${SQLLM_PACKED_PATH}")" "${EVAL_CACHE_DIR}/packed/$(basename "${SQLLM_PACKED_PATH}")"
  ln -sfn "$(absolute_path "${LNQ_PACKED_PATH}")" "${EVAL_CACHE_DIR}/layerwise_packed/$(basename "${LNQ_PACKED_PATH}")"
}

run_repo_eval() {
  prepare_eval_cache

  local eval_args=(
    run_eval.py
    --cache_dir "${EVAL_CACHE_DIR}"
    --output_file "${RESULTS_FILE}"
  )

  if [[ "${REDO_EVAL}" == "1" ]]; then
    eval_args+=(--redo)
  fi
  if [[ "${DOWNSTREAM}" == "1" ]]; then
    eval_args+=(--downstream)
  fi

  log "Running repo eval with 4096 context via ${RESULTS_FILE}"
  "${PYTHON_BIN}" "${eval_args[@]}"

  log "Results written to ${RESULTS_FILE}"
  "${PYTHON_BIN}" -m json.tool "${RESULTS_FILE}"
  cleanup_after_stage "evaluation"
}

main() {
  log "Reproduction target:"
  log "  Paper table: Table 3, weight-only scalar PTQ without end-to-end fine-tuning"
  log "  Expected 3-bit Llama-2-7B paper PPL: Original Wiki2=5.12 C4=6.63; SqueezeLLM Wiki2=5.74 C4=7.44; LNQ Wiki2=5.89 C4=7.74"
  log "  This script runs plain LNQ with --is_nosal true, not LNQ + GuidedQuant."

  assert_paper_config
  maybe_install_deps
  check_environment
  download_file "${TOKEN_URL}" "${TOKEN_PATH}"
  verify_calibration

  if [[ "${RUN_QUANTIZATION}" == "1" ]]; then
    clean_repro_outputs
    run_squeezellm
    run_lnq_plain
  else
    log "RUN_QUANTIZATION=0; using existing packed models"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    run_repo_eval
  else
    log "RUN_EVAL=0; skipping evaluation"
  fi
}

main "$@"
