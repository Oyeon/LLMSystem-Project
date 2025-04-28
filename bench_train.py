#!/usr/bin/env python
"""
One-epoch wall-clock + peak-VRAM benchmark for MAGDi variants.

  CUDA_VISIBLE_DEVICES=0 python bench_train.py --kind fused    --eightbit --checkpoint
  CUDA_VISIBLE_DEVICES=0 python bench_train.py --kind original --eightbit --checkpoint
"""
import os, argparse, json, pickle, gc, torch
from importlib import import_module
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from torch_geometric.data import Data
from timer_callback import Timer
import utils, data_utils


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.set_device(0)
torch.cuda.empty_cache(); gc.collect()

BACKBONE  = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET   = "ARC"
MAX_NODES = 12

# ───────────────────────── dataset helper ────────────────────────────
def get_dataset(n_samples, seq_len, tok):
    try:
        node_emb = pickle.loads(open(f"node_emb/{DATASET}_node_emb.pkl", "rb").read())
        with open(f"MAG/{DATASET}_1000.json") as f:
            all_res = json.load(f)[:n_samples]
        node_emb = torch.tensor(node_emb.reshape(-1, MAX_NODES,
                                                 node_emb.shape[-1])[:n_samples])
        graphs = utils.construct_graphs(all_res, node_emb, n_samples, MAX_NODES)
        train_b, graphs = utils.pad_graphs(
            utils.prepare_batch(tok, all_res, n_samples, MAX_NODES), graphs)
    except FileNotFoundError:
        print("⚠️  ARC files missing – using synthetic data.")
        graphs = [Data(
            x=torch.randn(MAX_NODES, 4096),
            edge_index=torch.stack([torch.arange(MAX_NODES-1),
                                    torch.arange(1, MAX_NODES)]),
            y=torch.randint(0, 3, (MAX_NODES,))
        ) for _ in range(n_samples)]
        idsA = tok("A"*seq_len, return_tensors="pt").input_ids[0]
        idsB = tok("B"*seq_len, return_tensors="pt").input_ids[0]
        attn = torch.ones_like(idsA)
        train_b = [{"pos_input_ids": idsA, "pos_attention_mask": attn,
                    "pos_labels": idsA,
                    "neg_input_ids": idsB, "neg_attention_mask": attn,
                    "neg_labels": idsB,
                    "graph_idx": i} for i in range(n_samples)]
    return train_b, data_utils.MAGDiDataCollator(tok, graphs)

# ───────────────────────── model factory ─────────────────────────────
# ───────── model factory ────────────────────────────────────────────
def build_magdi(kind, eightbit=False, use_gc=False):
    mod = import_module("model_fused" if kind == "fused" else "model_original")

    q_cfg = BitsAndBytesConfig(load_in_8bit=True) if eightbit else None

    mdl = mod.MAGDi(
        BACKBONE, 4096, 512, 3, 1, 1, 0.1,
        quantization_config=q_cfg,              # None for fp16
        device_map="auto" if eightbit else None,
        use_gc=use_gc,                          # ← keep only **one** copy
    )

    # 8-bit models are already on the right device; fp16 ones need .cuda()
    return mdl if eightbit else mdl.cuda()

# ───────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["fused", "original"])
    ap.add_argument("--bs",   type=int, default=1)
    ap.add_argument("--seq",  type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eightbit",   action="store_true")
    ap.add_argument("--checkpoint", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        BACKBONE, use_fast=False, legacy=True,
        padding_side="left", add_eos_token=True)
    tok.pad_token_id = tok.eos_token_id

    train_ds, coll = get_dataset(args.bs * args.grad_accum, args.seq, tok)
    model = build_magdi(args.kind, eightbit=args.eightbit, use_gc=args.checkpoint)

    TrainerCls = import_module(
        "model_fused" if args.kind == "fused" else "model_original"
    ).MAGDiTrainer

    trainer = TrainerCls(
        model=model,
        train_dataset=train_ds,
        data_collator=coll,
        args=TrainingArguments(
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            fp16=True,
            max_grad_norm=1.0,
            logging_steps=10,
            output_dir=f"out_{args.kind}",
            save_strategy="no",
            remove_unused_columns=False,
        ),
        callbacks=[Timer()],
    )
    trainer.train()
