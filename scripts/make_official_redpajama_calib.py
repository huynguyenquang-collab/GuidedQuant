#!/usr/bin/env python3
import argparse
import os
import random

import torch
from datasets import Features, Value, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a RedPajama calibration token cache from a HF dataset."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--dataset", default="ZengXiangyu/RedPajama-Data-1T-Sample")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    return parser.parse_args()


def redpajama_features(dataset_name, config_name):
    if dataset_name != "togethercomputer/RedPajama-Data-1T":
        return None

    # The official c4 config stores "meta" as a struct in the data files. Some
    # versions of datasets/custom-code metadata try to cast it to string while
    # streaming, which fails before any row is yielded. Override that schema.
    if config_name == "c4":
        return Features(
            {
                "text": Value("string"),
                "meta": {
                    "timestamp": Value("string"),
                    "url": Value("string"),
                    "language": Value("string"),
                    "source": Value("string"),
                },
                "red_pajama_subset": Value("string"),
            }
        )

    return None


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    use_streaming = args.dataset == "togethercomputer/RedPajama-Data-1T"
    print(
        "Loading RedPajama calibration source: "
        f"dataset={args.dataset}, config={args.config or '<none>'}, split={args.split}, streaming={use_streaming}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    features = redpajama_features(args.dataset, args.config)
    if features is not None:
        print(f"Using explicit features override for {args.dataset}/{args.config}", flush=True)

    load_args = [args.dataset]
    if args.config:
        load_args.append(args.config)
    dataset = load_dataset(
        *load_args,
        split=args.split,
        streaming=use_streaming,
        trust_remote_code=True,
        features=features,
    )
    if use_streaming:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    tokens = []
    seen = 0
    progress = tqdm(total=args.num_examples, desc=f"Making RedPajama {args.num_examples}x{args.seq_len}")

    if use_streaming:
        for item in dataset:
            seen += 1
            text = item.get(args.text_field)
            if text is None:
                text = next((value for value in item.values() if isinstance(value, str)), None)
            if not isinstance(text, str) or not text.strip():
                continue

            encoded = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
            if encoded.numel() < args.seq_len:
                continue

            start = rng.randint(0, encoded.numel() - args.seq_len)
            tokens.append(encoded[start : start + args.seq_len].to(torch.long))
            progress.update(1)
            if len(tokens) >= args.num_examples:
                break
    else:
        selected_indices = set()
        while len(tokens) < args.num_examples:
            seen += 1
            idx = rng.randint(0, len(dataset) - 1)
            if idx in selected_indices:
                continue

            item = dataset[idx]
            text = item.get(args.text_field)
            if text is None:
                text = next((value for value in item.values() if isinstance(value, str)), None)
            if not isinstance(text, str) or not text.strip():
                continue

            encoded = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
            if encoded.numel() < args.seq_len:
                continue

            start = rng.randint(0, encoded.numel() - args.seq_len)
            tokens.append(encoded[start : start + args.seq_len].to(torch.long))
            selected_indices.add(idx)
            progress.update(1)
    progress.close()

    if len(tokens) != args.num_examples:
        raise SystemExit(
            f"Only collected {len(tokens)} calibration samples after scanning {seen} documents; "
            f"expected {args.num_examples}. Try a larger --shuffle-buffer or another official RedPajama text config."
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tmp_output = f"{args.output}.tmp"
    torch.save(tokens, tmp_output)
    os.replace(tmp_output, args.output)
    print(
        f"saved={args.output} type=list len={len(tokens)} sample_shape={tuple(tokens[0].shape)} scanned={seen}",
        flush=True,
    )


if __name__ == "__main__":
    main()
