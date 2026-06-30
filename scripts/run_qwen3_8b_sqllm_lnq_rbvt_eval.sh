#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS="${MODELS:-Qwen3_8B}" exec bash "${SCRIPT_DIR}/run_mistral_qwen_sqllm_lnq_rbvt_eval.sh"
