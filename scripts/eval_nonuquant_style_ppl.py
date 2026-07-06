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
def evaluate_sliding_window(
    model,
    tokenizer,
    texts,
    device: str,
    max_length: int,
    stride: int,
    limit_tokens: int | None,
    batch_size: int,
):
    model.eval()
    total_nll = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = 0
    total_eval_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

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
        windows = []
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            windows.append((begin_loc, end_loc, trg_len))
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar = tqdm(range(0, len(windows), batch_size), desc=f"Windows ({seq_len:,} toks)", unit="batch", leave=False)
        for start in pbar:
            batch_windows = windows[start : start + batch_size]
            batch_input = torch.full((len(batch_windows), max_length), pad_token_id, dtype=input_ids.dtype, device=device)
            batch_labels = torch.full_like(batch_input, -100)
            batch_attention = torch.zeros_like(batch_input)

            for row, (begin_loc, end_loc, trg_len) in enumerate(batch_windows):
                input_chunk = input_ids[:, begin_loc:end_loc]
                chunk_len = input_chunk.size(1)
                batch_input[row, :chunk_len] = input_chunk[0]
                batch_attention[row, :chunk_len] = 1
                target_chunk = input_chunk.clone()
                if begin_loc > 0:
                    target_chunk[:, :-trg_len] = -100
                batch_labels[row, :chunk_len] = target_chunk[0]

            outputs = model(batch_input, attention_mask=batch_attention, use_cache=False)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch_labels[:, 1:].contiguous()
            loss_sum = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            label_count = shift_labels.ne(-100).sum()
            total_nll += loss_sum.float()
            total_eval_tokens += int(label_count.item())

            denom = max(total_eval_tokens, 1)
            current_ppl = torch.exp(total_nll / denom).item()
            pbar.set_postfix({"PPL": f"{current_ppl:.4f}", "tokens": f"{total_tokens + batch_windows[-1][1]:,}"})

        total_tokens += seq_len

    if total_eval_tokens == 0:
        raise RuntimeError("No valid tokens were evaluated.")

    return {
        "perplexity": torch.exp(total_nll / total_eval_tokens).item(),
        "total_tokens": total_tokens,
        "eval_tokens": total_eval_tokens,
    }


@torch.no_grad()
def evaluate_chunked(model, tokenizer, texts, device: str, max_length: int, limit_tokens: int | None, batch_size: int):
    model.eval()
    total_nll = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = 0
    total_eval_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

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
        nsamples = seq_len // max_length
        if nsamples == 0:
            continue

        chunks = input_ids[:, : nsamples * max_length].view(nsamples, max_length)
        pbar = tqdm(range(0, nsamples, batch_size), desc=f"Chunks ({seq_len:,} toks)", unit="batch", leave=False)
        for start in pbar:
            batch_input = chunks[start : start + batch_size]
            batch_attention = torch.ones_like(batch_input)
            outputs = model(batch_input, attention_mask=batch_attention, use_cache=False)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch_input[:, 1:].contiguous()
            loss_sum = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            label_count = shift_labels.numel()
            total_nll += loss_sum.float()
            total_eval_tokens += int(label_count)
            pbar.set_postfix({"PPL": f"{torch.exp(total_nll / total_eval_tokens).item():.4f}", "tokens": f"{total_tokens + (start + batch_input.size(0)) * max_length:,}"})

        total_tokens += nsamples * max_length

    if total_eval_tokens == 0:
        raise RuntimeError("No valid tokens were evaluated.")

    return {
        "perplexity": torch.exp(total_nll / total_eval_tokens).item(),
        "total_tokens": total_tokens,
        "eval_tokens": total_eval_tokens,
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-mode",
        default="sliding",
        choices=["sliding", "chunks"],
        help="sliding keeps NonUQuantFix window/stride; chunks matches SqueezeLLM/GPTQ non-overlap eval.",
    )
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

            print(f"\nEvaluating {args.model_name or Path(args.model_path).name} on {dataset_name} ({args.eval_mode})")
            if args.eval_mode == "chunks":
                results[dataset_name] = evaluate_chunked(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    device=args.device,
                    max_length=args.max_length,
                    limit_tokens=args.limit_tokens or None,
                    batch_size=args.batch_size,
                )
            else:
                results[dataset_name] = evaluate_sliding_window(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    device=args.device,
                    max_length=args.max_length,
                    stride=args.stride,
                    limit_tokens=args.limit_tokens or None,
                    batch_size=args.batch_size,
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
        "eval_mode": args.eval_mode,
        "batch_size": args.batch_size,
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
