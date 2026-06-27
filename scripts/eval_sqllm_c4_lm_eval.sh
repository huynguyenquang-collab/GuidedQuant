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
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
MODEL_PATH="${MODEL_PATH:-cache/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-c4_s${NUM_EXAMPLES}_blk${SEQ_LEN}}"
OUTPUT_FILE="${OUTPUT_FILE:-results_sqllm_c4_lm_eval_${MODEL_BASENAME}_${BITS}bit.json}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_easy arc_challenge hellaswag piqa winogrande boolq rte openbookqa lambada_openai mmlu gsm8k}"

python - "$MODEL_PATH" "$OUTPUT_FILE" "$LM_EVAL_BATCH_SIZE" "$LM_EVAL_NUM_FEWSHOT" "$LM_EVAL_LIMIT" $LM_EVAL_TASKS <<'PY'
import json
import inspect
import sys

import lm_eval

from any_precision.evaluate import eval

model_path = sys.argv[1]
output_file = sys.argv[2]
batch_size = sys.argv[3]
num_fewshot = None if sys.argv[4] == "" else int(sys.argv[4])
limit = None if sys.argv[5] == "" else float(sys.argv[5])
tasks = sys.argv[6:]

tokenizer_type, tokenizer, model = eval.auto_model_load(model_path)
model.eval()

supported_bits = model.precisions if hasattr(model, "precisions") else [None]
results = {}
for bit in supported_bits:
    if bit is not None:
        print(f"<<<< Setting model precision to {bit}-bit... >>>>")
        model.set_precision(bit)

    try:
        model_lm = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
    except TypeError:
        print("HFLM does not accept batch_size in this lm_eval version; falling back without it.")
        model_lm = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
        )

    simple_evaluate_params = inspect.signature(lm_eval.simple_evaluate).parameters
    eval_kwargs = {
        "model": model_lm,
        "tasks": tasks,
    }
    if num_fewshot is not None and "num_fewshot" in simple_evaluate_params:
        eval_kwargs["num_fewshot"] = num_fewshot
    elif num_fewshot is not None:
        print("simple_evaluate does not accept num_fewshot in this lm_eval version; ignoring it.")
    if limit is not None and "limit" in simple_evaluate_params:
        eval_kwargs["limit"] = limit
    elif limit is not None:
        print("simple_evaluate does not accept limit in this lm_eval version; ignoring it.")
    if "batch_size" in simple_evaluate_params:
        eval_kwargs["batch_size"] = batch_size

    eval_results = lm_eval.simple_evaluate(**eval_kwargs)
    key = f"{bit}-bit" if bit is not None else "fp"
    results[key] = eval_results

with open(output_file, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "model_path": model_path,
            "tasks": tasks,
            "batch_size": batch_size,
            "num_fewshot": num_fewshot,
            "limit": limit,
            "lm_eval": results,
        },
        handle,
        indent=2,
        sort_keys=True,
        default=str,
    )

print(json.dumps({k: v.get("results", {}) for k, v in results.items()}, indent=2, sort_keys=True, default=str))
print(f"Saved lm_eval results to {output_file}")
PY
