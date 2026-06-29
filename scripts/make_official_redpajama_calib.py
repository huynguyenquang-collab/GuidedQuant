#!/usr/bin/env python3
import argparse
import os
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a RedPajama calibration token cache from togethercomputer/RedPajama-Data-1T."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--dataset", default="togethercomputer/RedPajama-Data-1T")
    parser.add_argument("--config", default="c4")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print(
        "Loading official RedPajama calibration source: "
        f"dataset={args.dataset}, config={args.config}, split={args.split}, streaming=True",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    tokens = []
    seen = 0
    progress = tqdm(total=args.num_examples, desc=f"Making official RedPajama {args.num_examples}x{args.seq_len}")
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
