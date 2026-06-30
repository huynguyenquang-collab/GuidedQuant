#!/usr/bin/env python3
"""Apply RBVT to GuidedQuant/SqueezeLLM LUT assignments and repack."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from any_precision.analyzer import dispatch_model, get_analyzer
from any_precision.quantization.pack import pack


@dataclass
class RBVTStats:
    flips: int = 0
    channels: int = 0
    candidates: int = 0
    boundary_kept: int = 0
    bias_before: float = 0.0
    bias_after: float = 0.0
    objective_before: float = 0.0
    objective_after: float = 0.0
    variance_increase: float = 0.0

    def add(self, other: "RBVTStats"):
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


class ActivationStatsCollector:
    def __init__(self, analyzer, want_var: bool):
        self.analyzer = analyzer
        self.want_var = want_var
        self.sum: dict[str, torch.Tensor] = {}
        self.sumsq: dict[str, torch.Tensor] = {}
        self.count: dict[str, int] = {}
        self.hooks = []

    def _hook(self, name: str):
        def hook(_module, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.detach().reshape(-1, x.shape[-1]).float()
            s = x.sum(dim=0).cpu()
            if name not in self.sum:
                self.sum[name] = s
                self.count[name] = x.shape[0]
                if self.want_var:
                    self.sumsq[name] = (x * x).sum(dim=0).cpu()
            else:
                self.sum[name] += s
                self.count[name] += x.shape[0]
                if self.want_var:
                    self.sumsq[name] += (x * x).sum(dim=0).cpu()
        return hook

    def register(self):
        for layer_idx, layer in enumerate(self.analyzer.get_layers()):
            for module_name, module in self.analyzer.get_modules(layer).items():
                name = f"{layer_idx:02}={module_name}"
                self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def means_vars(self):
        means = {}
        variances = {}
        for name, total in self.sum.items():
            count = max(1, self.count[name])
            mean = total / count
            means[name] = mean
            if self.want_var and name in self.sumsq:
                ex2 = self.sumsq[name] / count
                variances[name] = (ex2 - mean * mean).clamp(min=0.0)
        return means, variances


def collect_activation_stats(analyzer, tokens: torch.Tensor, n_calib: int, batch_size: int, rbvt_lambda: float):
    model = analyzer.model
    if torch.cuda.device_count() > 1:
        model = dispatch_model(model)
    model = model.bfloat16()
    model.eval()
    if model.device.type != "cuda" and torch.cuda.device_count() == 1:
        model.cuda()

    collector = ActivationStatsCollector(analyzer, want_var=rbvt_lambda > 0.0)
    collector.register()
    try:
        limit = min(n_calib, tokens.shape[0])
        for start in tqdm(range(0, limit, batch_size), desc="Collecting RBVT activation stats"):
            batch = tokens[start:start + batch_size].to(model.device)
            with torch.inference_mode():
                model(input_ids=batch, use_cache=False)
            del batch
            if torch.cuda.is_available() and (start // batch_size + 1) % 16 == 0:
                torch.cuda.empty_cache()
    finally:
        collector.remove()

    model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return collector.means_vars()


def normalize_tokens(tokens, seq_len: int) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim == 2:
            return tokens.long()
        if tokens.ndim == 3 and tokens.shape[1] == 1:
            return tokens[:, 0, :].long()
        raise ValueError(f"Expected token tensor with shape [n, seq] or [n, 1, seq], got {tuple(tokens.shape)}")

    if isinstance(tokens, (list, tuple)):
        normalized = []
        for item in tokens:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"Expected tensor token item, got {type(item).__name__}")
            item = item.detach().cpu()
            if item.ndim == 2 and item.shape[0] == 1:
                item = item[0]
            if item.ndim != 1:
                raise ValueError(f"Expected token item with shape [seq] or [1, seq], got {tuple(item.shape)}")
            if item.numel() != seq_len:
                raise ValueError(f"Expected token length {seq_len}, got {item.numel()}")
            normalized.append(item.long())
        if not normalized:
            raise ValueError("Token list is empty")
        return torch.stack(normalized, dim=0)

    raise TypeError(f"Unsupported token cache type: {type(tokens).__name__}")


def _dequantize(indices: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    # indices: [rows, 1, cols], luts: [rows, 1, levels]
    row_ids = torch.arange(indices.shape[0], device=indices.device).view(-1, 1)
    return luts[:, 0, :][row_ids, indices[:, 0, :].long()]


@torch.no_grad()
def apply_rbvt_indices(
    W_fp: torch.Tensor,
    indices: torch.Tensor,
    luts: torch.Tensor,
    mu: torch.Tensor,
    sigma_ii: torch.Tensor | None,
    rbvt_lambda: float,
    rbvt_topk: int,
    row_chunk: int,
    gap_floor: float,
    strict_descent: bool,
) -> tuple[torch.Tensor, RBVTStats]:
    device = W_fp.device
    out_features, in_features = W_fp.shape
    levels = luts.to(device).float()
    idx_full = indices.to(device).long()[:, 0, :].clone()
    Wq_full = _dequantize(indices.to(device), levels).float()
    mu = mu.to(device).float()
    sigma_ii = torch.zeros_like(mu) if sigma_ii is None else sigma_ii.to(device).float()

    stats = RBVTStats(channels=out_features)
    num_levels = levels.shape[-1]
    relax_eps = 1e-12

    for r0 in range(0, out_features, row_chunk):
        r1 = min(r0 + row_chunk, out_features)
        rc = r1 - r0
        Wr = W_fp[r0:r1].float()
        Wq = Wq_full[r0:r1]
        idx = idx_full[r0:r1]
        row_ids = torch.arange(rc, device=device).unsqueeze(1)

        e = Wq - Wr
        e_sign = torch.sign(e)
        b = e @ mu

        left_idx = (idx - 1).clamp(min=0)
        right_idx = (idx + 1).clamp(max=num_levels - 1)
        cur = levels[r0:r1, 0, :][row_ids, idx]
        left = levels[r0:r1, 0, :][row_ids, left_idx]
        right = levels[r0:r1, 0, :][row_ids, right_idx]

        move_down = e_sign > 0
        gap = torch.where(move_down, (cur - left).abs(), (right - cur).abs())
        target_idx = torch.where(move_down, left_idx, right_idx)
        feasible = torch.where(move_down, idx > 0, idx < (num_levels - 1))
        gap_ok = gap > gap_floor

        v = mu.unsqueeze(0) * e_sign * gap
        r = v.abs()
        q = sigma_ii.unsqueeze(0) * (gap.square() - 2.0 * gap * e.abs()).clamp(min=0.0)
        sign_aligned = (b.unsqueeze(1) * v) > 0
        admissible = feasible & gap_ok & sign_aligned & (r > relax_eps)
        rho = q / (r + relax_eps)

        for rr in range(rc):
            T = float(abs(b[rr].item()))
            base_obj = T * T
            stats.bias_before += base_obj
            stats.objective_before += base_obj
            if T <= relax_eps:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            cand = torch.nonzero(admissible[rr], as_tuple=False).squeeze(1)
            stats.candidates += int(cand.numel())
            if cand.numel() == 0:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            cand_rho = rho[rr, cand]
            if rbvt_topk > 0 and cand.numel() > rbvt_topk:
                _, topk_idx = torch.topk(cand_rho, k=rbvt_topk, largest=False, sorted=False)
                cand = cand[topk_idx]
                cand_rho = cand_rho[topk_idx]
            cand = cand[torch.argsort(cand_rho, descending=False)]

            r_cand = r[rr, cand]
            q_cand = q[rr, cand]
            limit = T if strict_descent else 2.0 * T
            cum_r = torch.cumsum(r_cand, dim=0)
            cum_q = torch.cumsum(q_cand, dim=0)
            zero = torch.zeros(1, device=device, dtype=r_cand.dtype)
            s_prev = torch.cat([zero, cum_r[:-1]], dim=0)
            q_prev = torch.cat([zero, cum_q[:-1]], dim=0)

            upper = ((limit - s_prev) / (r_cand + relax_eps)).clamp(min=0.0, max=1.0)
            gamma_star = (T - s_prev - rbvt_lambda * q_cand / (2.0 * (r_cand + relax_eps))) / (r_cand + relax_eps)
            gamma = torch.minimum(torch.maximum(gamma_star, torch.zeros_like(gamma_star)), upper)
            relaxed_obj = (T - s_prev - gamma * r_cand).square() + rbvt_lambda * (q_prev + gamma * q_cand)
            relaxed_obj = torch.where(upper > 0.0, relaxed_obj, torch.full_like(relaxed_obj, float("inf")))

            best_val, best_pos = relaxed_obj.min(dim=0)
            if float(best_val.item()) >= base_obj:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            best_pos_i = int(best_pos.item())
            best_gamma = float(gamma[best_pos_i].item())
            prefix_count = best_pos_i
            prefix_r = float(s_prev[best_pos_i].item())
            prefix_q = float(q_prev[best_pos_i].item())
            drop_obj = (T - prefix_r) ** 2 + rbvt_lambda * prefix_q

            keep_obj = float("inf")
            keep_count = prefix_count
            keep_r = prefix_r
            keep_q = prefix_q
            if best_gamma > 0.0:
                keep_r = float((prefix_r + r_cand[best_pos_i]).item())
                if keep_r <= limit + 1e-8:
                    keep_q = float((prefix_q + q_cand[best_pos_i]).item())
                    keep_obj = (T - keep_r) ** 2 + rbvt_lambda * keep_q
                    keep_count = prefix_count + 1

            if keep_obj < drop_obj:
                chosen_count = keep_count
                stats.boundary_kept += int(best_gamma < 1.0)
                final_r = keep_r
                final_q = keep_q
            else:
                chosen_count = prefix_count
                final_r = prefix_r
                final_q = prefix_q

            if chosen_count > 0:
                chosen = cand[:chosen_count]
                idx_full[r0 + rr, chosen] = target_idx[rr, chosen]
                stats.flips += chosen_count

            stats.bias_after += (T - final_r) ** 2
            stats.objective_after += (T - final_r) ** 2 + rbvt_lambda * final_q
            stats.variance_increase += final_q

    return idx_full.cpu().to(torch.uint8).unsqueeze(1), stats


def apply_rbvt_to_sqllm_cache(args, analyzer, means, variances):
    src = Path(args.input_quantized_path)
    dst = Path(args.output_quantized_path)
    if args.overwrite and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for subdir in src.iterdir():
        if subdir.is_dir() and subdir.name.startswith("lut_"):
            shutil.copytree(subdir, dst / subdir.name, dirs_exist_ok=True)
    if (src / "misc_weights").exists():
        shutil.copytree(src / "misc_weights", dst / "misc_weights", dirs_exist_ok=True)
    (dst / "weights").mkdir(parents=True, exist_ok=True)

    total = RBVTStats()
    for layer_idx in tqdm(range(analyzer.num_layers), desc="Applying RBVT to SqueezeLLM cache"):
        output_weight_path = dst / "weights" / f"l{layer_idx}.pt"
        if output_weight_path.exists() and not args.overwrite:
            logging.info("Skipping completed RBVT layer cache: %s", output_weight_path)
            continue

        layer_weights = torch.load(src / "weights" / f"l{layer_idx}.pt", map_location="cpu")
        layer_luts = torch.load(src / f"lut_{args.bits}" / f"l{layer_idx}.pt", map_location="cpu")
        fp_weights = analyzer.get_layer_weights(layer_idx)
        out_layer = {}
        for module_name in analyzer.get_layer_module_names(layer_idx):
            stat_name = f"{layer_idx:02}={module_name}"
            if stat_name not in means:
                raise RuntimeError(f"Missing RBVT activation stats for {stat_name}")
            W_fp = fp_weights[module_name].to(args.device)
            indices = torch.as_tensor(layer_weights[module_name], device=args.device)
            luts = torch.as_tensor(layer_luts[module_name], device=args.device)
            new_indices, stats = apply_rbvt_indices(
                W_fp=W_fp,
                indices=indices,
                luts=luts,
                mu=means[stat_name],
                sigma_ii=variances.get(stat_name),
                rbvt_lambda=args.rbvt_lambda,
                rbvt_topk=args.rbvt_topk,
                row_chunk=args.row_chunk,
                gap_floor=args.gap_floor,
                strict_descent=not args.allow_overshoot,
            )
            out_layer[module_name] = new_indices.numpy()
            total.add(stats)
            del W_fp, indices, luts, new_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        torch.save(out_layer, output_weight_path)

    return total


def parse_args():
    parser = argparse.ArgumentParser(description="RBVT correction for GuidedQuant SqueezeLLM caches.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--dataset", default="redpajama")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--tokens-path", default="")
    parser.add_argument("--input-quantized-path", default="")
    parser.add_argument("--output-quantized-path", default="")
    parser.add_argument("--output-packed-path", default="")
    parser.add_argument("--stats-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-calib", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rbvt-lambda", type=float, default=1.0)
    parser.add_argument("--rbvt-topk", type=int, default=0)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--gap-floor", type=float, default=1e-8)
    parser.add_argument("--allow-overshoot", action="store_true")
    parser.add_argument("--cpu-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-stats", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s | %(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    model_basename = args.model.rstrip("/").split("/")[-1]
    run_name = f"{model_basename}-w{args.bits}-rbvt-sqllm-{args.dataset}_s{args.num_examples}_blk{args.seq_len}_lambda{args.rbvt_lambda:g}"
    args.tokens_path = args.tokens_path or f"{args.cache_dir}/tokens/{model_basename}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}.pt"
    args.input_quantized_path = args.input_quantized_path or f"{args.cache_dir}/quantized/{model_basename}-w{args.bits}_orig{args.bits}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}"
    args.output_quantized_path = args.output_quantized_path or f"{args.cache_dir}/rbvt_sqllm_quantized/{run_name}"
    args.output_packed_path = args.output_packed_path or f"{args.cache_dir}/rbvt_sqllm_packed/anyprec-rbvt-sqllm-{model_basename}-w{args.bits}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}_lambda{args.rbvt_lambda:g}"
    args.stats_path = args.stats_path or f"{args.cache_dir}/rbvt_sqllm_stats/{run_name}_n{args.n_calib}.pt"

    if not Path(args.tokens_path).exists():
        raise FileNotFoundError(f"Missing calibration tokens: {args.tokens_path}")
    if not Path(args.input_quantized_path).exists():
        raise FileNotFoundError(f"Missing SqueezeLLM quantized cache: {args.input_quantized_path}")
    if Path(args.output_packed_path).exists() and any(Path(args.output_packed_path).iterdir()) and not args.overwrite:
        logging.info("Packed output already exists; use --overwrite to rebuild: %s", args.output_packed_path)
        return

    logging.info("Loading model/analyzer: %s", args.model)
    analyzer = get_analyzer(args.model, include_tokenizer=True)
    tokens = normalize_tokens(torch.load(args.tokens_path, map_location="cpu"), args.seq_len)
    logging.info("Loaded calibration tokens: shape=%s", tuple(tokens.shape))

    stats_path = Path(args.stats_path)
    if stats_path.exists() and not args.overwrite_stats:
        logging.info("Loading cached RBVT activation stats: %s", stats_path)
        payload = torch.load(stats_path, map_location="cpu")
        means = payload["means"]
        variances = payload["variances"]
    else:
        means, variances = collect_activation_stats(
            analyzer=analyzer,
            tokens=tokens,
            n_calib=args.n_calib,
            batch_size=args.batch_size,
            rbvt_lambda=args.rbvt_lambda,
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": args.model,
                "tokens_path": args.tokens_path,
                "n_calib": args.n_calib,
                "seq_len": args.seq_len,
                "rbvt_lambda": args.rbvt_lambda,
                "means": means,
                "variances": variances,
            },
            stats_path,
        )
        logging.info("Saved RBVT activation stats: %s", stats_path)
    logging.info("RBVT activation stats ready: means=%d variances=%d", len(means), len(variances))

    totals = apply_rbvt_to_sqllm_cache(args, analyzer, means, variances)
    logging.info(
        "RBVT summary | flips=%d candidates=%d boundary_kept=%d bias %.6e -> %.6e objective %.6e -> %.6e var_inc=%.6e",
        totals.flips,
        totals.candidates,
        totals.boundary_kept,
        totals.bias_before,
        totals.bias_after,
        totals.objective_before,
        totals.objective_after,
        totals.variance_increase,
    )

    analyzer.drop_original_weights()
    if Path(args.output_packed_path).exists() and args.overwrite:
        shutil.rmtree(args.output_packed_path)
    pack(
        analyzer=analyzer,
        lut_path=args.output_quantized_path,
        output_model_path=args.output_packed_path,
        seed_precision=args.bits,
        parent_precision=args.bits,
        cpu_count=args.cpu_count,
    )
    logging.info("RBVT-SqueezeLLM packed model saved to %s", args.output_packed_path)


if __name__ == "__main__":
    main()
