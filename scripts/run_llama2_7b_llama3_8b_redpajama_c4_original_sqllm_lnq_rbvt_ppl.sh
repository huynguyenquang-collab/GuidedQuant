#!/usr/bin/env bash
set -Eeuo pipefail

JOB_NAME="${JOB_NAME:-llama2-7b-llama3-8b-redpajama-c4-sqllm-lnq-rbvt}"
trap 'echo "[${JOB_NAME}] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

REDPAJAMA_SCRIPT="${SCRIPT_DIR}/run_llama2_7b_llama3_8b_redpajama_original_sqllm_lnq_rbvt_ppl.sh"
C4_SCRIPT="${SCRIPT_DIR}/run_llama2_7b_llama3_8b_c4_original_sqllm_lnq_rbvt_ppl.sh"

if [[ ! -x "${REDPAJAMA_SCRIPT}" ]]; then
  echo "Missing executable RedPajama script: ${REDPAJAMA_SCRIPT}" >&2
  exit 1
fi
if [[ ! -x "${C4_SCRIPT}" ]]; then
  echo "Missing executable C4 script: ${C4_SCRIPT}" >&2
  exit 1
fi

CALIBS="${CALIBS:-redpajama c4}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${JOB_NAME}] $*"
}

run_calib() {
  local calib="$1"
  case "${calib}" in
    redpajama|red|rpj)
      log "Starting RedPajama calibration job"
      bash "${REDPAJAMA_SCRIPT}"
      ;;
    c4)
      log "Starting C4 calibration job"
      bash "${C4_SCRIPT}"
      ;;
    *)
      echo "Unsupported calibration '${calib}'. Expected one of: redpajama c4" >&2
      exit 1
      ;;
  esac
}

for calib in ${CALIBS}; do
  run_calib "${calib}"
done

log "Done. Results:"
log "  RedPajama:"
log "    outputs/llama2_7b_3bit_redpajama_original_sqllm_lnq_rbvt/ppl"
log "    outputs/llama3_8b_3bit_redpajama_original_sqllm_lnq_rbvt/ppl"
log "  C4:"
log "    outputs/llama2_7b_3bit_c4_original_sqllm_lnq_rbvt/ppl"
log "    outputs/llama3_8b_3bit_c4_original_sqllm_lnq_rbvt/ppl"
