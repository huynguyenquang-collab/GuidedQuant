#!/usr/bin/env bash
set -euo pipefail
set -x

if [[ -n "${CUDA_DEVICE:-}" ]]; then
  export GUIDEDQUANT_CUDA_DEVICE="$CUDA_DEVICE"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
MODEL_BASENAME="${MODEL_NAME##*/}"
BITS="${BITS:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"
MODEL_PATH="${MODEL_PATH:-cache/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-c4_s${NUM_EXAMPLES}_blk${SEQ_LEN}}"
OUTPUT_FILE="${OUTPUT_FILE:-results_sqllm_c4_ppl_${MODEL_BASENAME}_${BITS}bit.json}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"

python - "$MODEL_PATH" "$OUTPUT_FILE" "$CHUNK_SIZE" <<'PY'
import json
import sys

from any_precision.evaluate import eval

model_path = sys.argv[1]
output_file = sys.argv[2]
chunk_size = int(sys.argv[3])
datasets = ["wikitext2", "c4"]

tokenizer_type, tokenizer, model = eval.auto_model_load(model_path)
results = eval.evaluate_ppl(
    model,
    tokenizer,
    datasets,
    verbose=True,
    chunk_size=chunk_size,
    tokenizer_type=tokenizer_type,
)

with open(output_file, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "model_path": model_path,
            "chunk_size": chunk_size,
            "ppl": results,
        },
        handle,
        indent=2,
        sort_keys=True,
    )

print(json.dumps(results, indent=2, sort_keys=True))
print(f"Saved PPL results to {output_file}")
PY
