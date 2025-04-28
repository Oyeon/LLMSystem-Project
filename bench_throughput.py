#!/usr/bin/env python
"""
Bench inference throughput (tokens / s) for MAGDi variants that
build their own decoder on the fly.

Usage
-----
CUDA_VISIBLE_DEVICES=0 python bench_throughput.py --kind fused
CUDA_VISIBLE_DEVICES=0 python bench_throughput.py --kind original
"""
import argparse, time, torch
from importlib import import_module
from transformers import AutoTokenizer

BACKBONE = "mistralai/Mistral-7B-Instruct-v0.2"

# ------------------------------------------------------------
def build_magdi(kind: str):
    mod   = import_module("model_fused" if kind == "fused" else "model_original")
    mdl   = mod.MAGDi(
        model_name          = BACKBONE,
        gcn_in_channels     = 4096,
        gcn_hidden_channels = 512,
        gcn_out_channels    = 3,
        alpha = 1, beta = 1, gamma = 0.1,
    )
    return mdl.cuda().eval()

# ------------------------------------------------------------
@torch.inference_mode()
def tokens_per_sec(model, batch, steps=100):
    # tiny warm-up
    for _ in range(10):
        _ = model.decoder(**batch)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = model.decoder(**batch)          # forward only
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return batch["input_ids"].numel() * steps / (t1 - t0)

# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["fused", "original"], required=True)
    ap.add_argument("--bs",   type=int, default=4)
    ap.add_argument("--seq",  type=int, default=256)
    ap.add_argument("--steps",type=int, default=100)
    args = ap.parse_args()

    tok   = AutoTokenizer.from_pretrained(BACKBONE)
    batch = tok(["x"*args.seq]*args.bs, return_tensors="pt").to("cuda")

    model = build_magdi(args.kind)
    tps   = tokens_per_sec(model, batch, args.steps)

    print(f"{args.kind.upper():>8}  {tps:,.0f} tokens/s  "
          f"(bs={args.bs}, seq={args.seq})")
