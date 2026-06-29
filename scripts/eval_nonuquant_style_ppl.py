#!/usr/bin/env python3
"""Evaluate GuidedQuant packed models with NonUQuantFix-style sliding-window PPL."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from any_precision.modules import AnyPrecisionForCausalLM


def _load_tokenizer(model_path: str, fallback_model: str | None, token: str | None):
    source = fallback_model or model_path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            use_fast=True,
            token=token,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_path: str, device: str, dtype: torch.dtype, precision: int | None, token: str | None):
    model_name = Path(model_path.rstrip("/")).name
    if model_name.startswith(("anyprec-", "layerwise-", "blockwise-", "full-")):
        model = AnyPrecisionForCausalLM.from_quantized(model_path, torch_dtype=dtype)
        if precision is not None:
            model.set_precision(precision)
        return model

    device_map = {"": device} if device != "auto" else "auto"
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        token=token,
    )


def _load_wikitext2(cache_dir: Path, seed: int, token: str | None):
    cache_file = cache_dir / f"wikitext2_test_seed{seed}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        token=token,
    )
    texts = ["\n".join([text for text in dataset["text"] if text])]
    with open(cache_file, "wb") as handle:
        pickle.dump(texts, handle)
    return texts


def _load_c4(cache_dir: Path, seed: int, n_samples: int, token: str | None):
    cache_file = cache_dir / f"c4_validation_n{n_samples}_seed{seed}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
        token=token,
    )
    texts = []
    for item in tqdm(dataset, total=n_samples, desc="Collecting C4"):
        if len(texts) >= n_samples:
            break
        text = item["text"].strip()
        if len(text) > 500:
            texts.append(text)

    result = ["\n\n".join(texts)]
    with open(cache_file, "wb") as handle:
        pickle.dump(result, handle)
    return result


@torch.no_grad()
def evaluate_sliding_window(model, tokenizer, texts, device: str, max_length: int, stride: int, limit_tokens: int | None):
    model.eval()
    nlls = []
    total_tokens = 0

    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        if tokenizer.bos_token_id is not None:
            if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                bos = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                input_ids = torch.cat([bos, input_ids], dim=1)

        if limit_tokens is not None and input_ids.size(1) > limit_tokens:
            input_ids = input_ids[:, :limit_tokens]

        input_ids = input_ids.to(device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue

        prev_end_loc = 0
        window_range = range(0, seq_len, stride)
        pbar = tqdm(window_range, desc=f"Windows ({seq_len:,} toks)", unit="win", leave=False)

        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_chunk = input_ids[:, begin_loc:end_loc]
            target_chunk = input_chunk.clone()
            if begin_loc > 0:
                target_chunk[:, :-trg_len] = -100

            outputs = model(input_chunk, labels=target_chunk)
            nlls.append(outputs.loss * trg_len)
            prev_end_loc = end_loc

            current_nll = torch.stack(nlls).sum()
            current_ppl = torch.exp(current_nll / (total_tokens + prev_end_loc)).item()
            pbar.set_postfix({"PPL": f"{current_ppl:.4f}", "tokens": f"{total_tokens + prev_end_loc:,}"})

            if end_loc == seq_len:
                break

        total_tokens += seq_len

    if not nlls:
        raise RuntimeError("No valid tokens were evaluated.")

    total_nll = torch.stack(nlls).sum()
    return {
        "perplexity": torch.exp(total_nll / total_tokens).item(),
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GuidedQuant models with NonUQuantFix sliding-window perplexity."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--tokenizer-path", default="", help="Fallback tokenizer, e.g. meta-llama/Llama-2-7b-hf.")
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"], choices=["wikitext2", "c4"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--c4-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./dataset_cache_nonuquant")
    parser.add_argument("--limit-tokens", type=int, default=0, help="Debug cap; 0 means no cap.")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args.model_path, args.tokenizer_path or None, args.hf_token)
    model = _load_model(args.model_path, args.device, dtype, args.precision, args.hf_token)

    results = {}
    try:
        for dataset_name in args.datasets:
            if dataset_name == "wikitext2":
                texts = _load_wikitext2(cache_dir, args.seed, args.hf_token)
            else:
                texts = _load_c4(cache_dir, args.seed, args.c4_samples, args.hf_token)

            print(f"\nEvaluating {args.model_name or Path(args.model_path).name} on {dataset_name}")
            results[dataset_name] = evaluate_sliding_window(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=args.device,
                max_length=args.max_length,
                stride=args.stride,
                limit_tokens=args.limit_tokens or None,
            )
            print(
                f"{dataset_name}: ppl={results[dataset_name]['perplexity']:.6f} "
                f"tokens={results[dataset_name]['total_tokens']:,}"
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    payload = {
        "model_path": args.model_path,
        "model_name": args.model_name or Path(args.model_path).name,
        "precision": args.precision,
        "stride": args.stride,
        "max_length": args.max_length,
        "c4_samples": args.c4_samples,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
