# --------------------------------------------------------------------
#   python train.py --dataset ARC --map
#   python train.py --dataset ARC --compile
# --------------------------------------------------------------------

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, TrainingArguments
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from peft import LoraConfig, get_peft_model

import utils, data_utils
from model import MAGDi, MAGDiTrainer 

np.random.seed(0)


def build_magdi(args):
    model = MAGDi(
        model_name=args.model_name,
        gcn_in_channels=args.gcn_in,
        gcn_hidden_channels=args.gcn_hidden,
        gcn_out_channels=args.gcn_out,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    # freeze base decoder
    for p in model.decoder.parameters():
        p.requires_grad_(False)
        if p.ndim == 1:
            p.data = p.data.to(torch.float32)

    # plug LoRA
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model.decoder.enable_input_require_grads()
    model.decoder.gradient_checkpointing_enable()
    model.decoder = get_peft_model(model.decoder, lora_cfg)
    model.decoder.lm_head = utils.CastOutputToFloat(model.decoder.lm_head)

    # optional graph-compiler (Tier-2)
    if args.compile:
        from torch import _dynamo as dynamo      # <-- no local 'torch'
        dynamo.config.cache_size_limit = 64
        model = torch.compile(model, mode="max-autotune")

    # optional multi-GPU sharding (skip if compiled)
    if args.map and not args.compile:
        max_mem = get_balanced_memory(
            model,
            no_split_module_classes=["GCN", "MistralDecoderLayer"],
            dtype="float16",
        )
        dmap = infer_auto_device_map(
            model,
            max_memory=max_mem,
            no_split_module_classes=["GCN", "MistralDecoderLayer"],
            dtype="float16",
        )
        model = dispatch_model(model, device_map=dmap)

    return model


# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ARC",
                    choices=["SQA", "ECQA", "ARC", "GSM8K", "MATH"])
    ap.add_argument("--model_name",
                    default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--gcn_in", type=int, default=4096)
    ap.add_argument("--gcn_hidden", type=int, default=512)
    ap.add_argument("--gcn_out", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--train_samples", type=int, default=1000)
    ap.add_argument("--max_nodes", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--compile", action="store_true",
                    help="enable torch.compile (Tier-2); single GPU only")
    ap.add_argument("--map", action="store_true",
                    help="use Accelerate device_map (Tier-1 multi-GPU)")
    args = ap.parse_args()

    # ─── load data ──────────────────────────────────────────────
    node_emb = pickle.loads(
        Path(f"node_emb/{args.dataset}_node_emb.pkl").read_bytes())
    with open(f"MAG/{args.dataset}_1000.json") as f:
        all_res = json.load(f)[: args.train_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=False, legacy=True,
        padding_side="left", add_eos_token=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # ─── prepare batches & graphs once ──────────────────────────
    node_emb = node_emb.reshape(
        args.train_samples, args.max_nodes, -1)[: args.train_samples]
    graphs = utils.construct_graphs(
        all_res, torch.tensor(node_emb), args.train_samples, args.max_nodes)
    train_batch = utils.prepare_batch(
        tokenizer, all_res, args.train_samples, args.max_nodes)
    train_batch, graphs = utils.pad_graphs(train_batch, graphs)

    # collator returns the pre-batched graph
    collator = data_utils.MAGDiDataCollator(tokenizer, graphs)

    # ─── build model ────────────────────────────────────────────
    model = build_magdi(args)

    print(f"✓ dataset {args.dataset} | samples {len(train_batch)}")

    trainer = MAGDiTrainer(
        model=model,
        train_dataset=train_batch,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=10,
            output_dir="outputs",
            remove_unused_columns=False,
            save_strategy="no",
        ),
        data_collator=collator,
    )

    trainer.train()
    model.decoder.save_pretrained(f"MAGDi_{args.dataset}")
    print("✓ training finished")


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()



###

# import json
# import torch
# import utils
# import data_utils
# import pickle
# import numpy as np
# np.random.seed(0)

# import networkx as nx
# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     AutoPeftModelForCausalLM
# )

# import argparse
# from model import MAGDi, MAGDiTrainer
# import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from accelerate import dispatch_model, infer_auto_device_map
# from accelerate.utils import get_balanced_memory



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # dataset: ['SQA', 'ECQA', 'ARC', 'GSM8K', 'MATH']
#     parser.add_argument('--dataset', default='MATH', type=str)
#     parser.add_argument('--model_name', default='mistralai/Mistral-7B-Instruct-v0.2', type=str)
#     parser.add_argument('--gcn_in_channels', default=4096, type=int)
#     parser.add_argument('--gcn_hidden_channels', default=512, type=int)
#     parser.add_argument('--gcn_out_channels', default=3, type=int)
#     parser.add_argument('--alpha', default=1.0, type=float)
#     parser.add_argument('--beta', default=1.0, type=float)
#     parser.add_argument('--gamma', default=0.1, type=float)
#     parser.add_argument('--num_train_samples', default=1000, type=int)
#     parser.add_argument('--max_node_num', default=12, type=int)    
#     parser.add_argument('--num_epochs', default=10, type=int)
#     parser.add_argument('--lr', default=5e-6, type=float)
#     args = parser.parse_args()

#     with open(f"node_emb/{args.dataset}_node_emb.pkl", "rb") as f:
#         node_embeddings = pickle.load(f)

#     with open(f"MAG/{args.dataset}_1000.json", "r") as f:
#         all_result = json.load(f)
#     all_result = all_result[:args.num_train_samples]

#     model = MAGDi(model_name=args.model_name,
#                 gcn_in_channels=args.gcn_in_channels,
#                 gcn_hidden_channels=args.gcn_hidden_channels,
#                 gcn_out_channels=args.gcn_out_channels,
#                 alpha=args.alpha,
#                 beta=args.beta,
#                 gamma=args.gamma)
#     model = torch.compile(model, mode="max-autotune")
                
#     node_embeddings = node_embeddings.reshape(args.num_train_samples, args.max_node_num, model.decoder.config.hidden_size)
#     node_embeddings = torch.tensor(node_embeddings)
#     node_embeddings = node_embeddings[:args.num_train_samples, :, :]
#     node_embeddings.size()

#     tokenizer = AutoTokenizer.from_pretrained(args.model_name,
#                                             use_fast=False,
#                                             legacy=True,
#                                             padding_side='left',
#                                             add_eos_token=True)
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     max_memory = get_balanced_memory(
#         model,
#         max_memory=None,
#         no_split_module_classes=["GCN", "MistralDecoderLayer"],
#         dtype='float16',
#         low_zero=False,
#     )

#     device_map = infer_auto_device_map(
#         model,
#         max_memory=max_memory,
#         no_split_module_classes=["GCN", "MistralDecoderLayer"],
#         dtype='float16'
#     )

#     model = dispatch_model(model, device_map=device_map)

#     for param in model.decoder.parameters():
#         param.requires_grad = False
#         if param.ndim == 1:
#             param.data = param.data.to(torch.float32)

#     config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )

#     model.decoder.gradient_checkpointing_enable()
#     model.decoder.enable_input_require_grads()
#     model.decoder = get_peft_model(model.decoder, config)
#     model.decoder.lm_head = utils.CastOutputToFloat(model.decoder.lm_head)
#     training_batch = utils.prepare_batch(tokenizer, all_result, args.num_train_samples, args.max_node_num)

#     graphs = utils.construct_graphs(all_result, node_embeddings, args.num_train_samples, args.max_node_num)
#     training_batch, graphs = utils.pad_graphs(training_batch, graphs)
#     print(len(training_batch), len(graphs))

#     trainer = MAGDiTrainer(
#         model=model, 
#         train_dataset=training_batch,
#         args=transformers.TrainingArguments(
#             per_device_train_batch_size=4, 
#             gradient_accumulation_steps=4,
#             warmup_steps=100, 
#             num_train_epochs=args.num_epochs,
#             learning_rate=args.lr,
#             fp16=True,
#             logging_steps=10, 
#             output_dir='outputs',
#             remove_unused_columns=False,
#             save_strategy="no"
#         ),
#         data_collator=data_utils.MAGDiDataCollator(tokenizer)
#     )

#     trainer.train()
#     model.decoder.save_pretrained("MAGDi_ARC")


######
# import os
# import json
# import torch
# import torch.nn as nn
# import pickle
# import numpy as np
# import argparse
# import re
# from collections import defaultdict

# np.random.seed(0)

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training
# )
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig
# )
# from accelerate import dispatch_model

# from model import MAGDi  
# import utils            
# import data_utils       # your data collator or loading utilities
# import networkx as nx

# torch.cuda.empty_cache()

# def train_one_epoch(model, dataloader, optimizer):
#     model.train()
#     total_loss = 0.0

#     # single "main" device if model is split across multiple GPUs
#     main_device = next(model.parameters()).device

#     for step, (batch, graph) in enumerate(dataloader):
#         # Move batch + graph to main_device
#         for k, v in batch.items():
#             if isinstance(v, torch.Tensor):
#                 batch[k] = v.to(main_device)
#         if hasattr(graph, 'to'):
#             graph = graph.to(main_device)

#         # Custom forward:
#         nll_loss, node_loss, mr_loss = model(
#             pos_input_ids=batch["pos_input_ids"],
#             pos_attention_mask=batch["pos_attention_mask"],
#             pos_labels=batch["pos_labels"],
#             neg_input_ids=batch["neg_input_ids"],
#             neg_attention_mask=batch["neg_attention_mask"],
#             neg_labels=batch["neg_labels"],
#             graph=graph
#         )
#         loss = nll_loss + node_loss + mr_loss

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         total_loss += loss.item()

#     return total_loss / len(dataloader)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default='SQA', type=str)
#     parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-Coder-V2-Lite-Base', type=str)
#     parser.add_argument('--gcn_in_channels', default=2048, type=int)
#     parser.add_argument('--gcn_hidden_channels', default=512, type=int)
#     parser.add_argument('--gcn_out_channels', default=3, type=int)
#     parser.add_argument('--alpha', default=1.0, type=float)
#     parser.add_argument('--beta', default=1.0, type=float)
#     parser.add_argument('--gamma', default=0.1, type=float)
#     parser.add_argument('--num_train_samples', default=1000, type=int)
#     parser.add_argument('--max_node_num', default=12, type=int)
#     parser.add_argument('--num_epochs', default=10, type=int)
#     parser.add_argument('--lr', default=5e-6, type=float)
#     parser.add_argument('--batch_size', default=1, type=int)
#     args = parser.parse_args()

#     # 1) Load data + embeddings
#     with open(f"node_emb/{args.dataset}_node_emb.pkl", "rb") as f:
#         node_embeddings = pickle.load(f)

#     with open(f"MAG/{args.dataset}_1000.json", "r") as f:
#         all_result = json.load(f)
#     all_result = all_result[:args.num_train_samples]

#     node_embeddings = node_embeddings.reshape(
#         args.num_train_samples,
#         args.max_node_num,
#         -1
#     )
#     node_embeddings = torch.tensor(node_embeddings, dtype=torch.float32)

#     # 2) BitsAndBytesConfig for 8-bit
#     bnb_config = BitsAndBytesConfig(
#         load_in_8bit=True,
#         bnb_8bit_use_memory_efficient_backward=False  # <--- crucial
#     )

#     # 3) Load base model in 8-bit
#     base_model_8bit = AutoModelForCausalLM.from_pretrained(
#         args.model_name,
#         trust_remote_code=True,
#         quantization_config=bnb_config,  # not load_in_8bit=...
#         device_map="auto"  # or "cpu"
#     )

#     # 4) Prepare for k-bit training
#     base_model_8bit = prepare_model_for_kbit_training(base_model_8bit)

#     # 5) Build MAGDi with that 8-bit base
#     model = MAGDi(
#         base_model=base_model_8bit,
#         gcn_in_channels=args.gcn_in_channels,
#         gcn_hidden_channels=args.gcn_hidden_channels,
#         gcn_out_channels=args.gcn_out_channels,
#         alpha=args.alpha,
#         beta=args.beta,
#         gamma=args.gamma
#     )

#     # Freeze base model weights
#     for param in model.decoder.parameters():
#         param.requires_grad = False

#     # 6) LoRA config + wrap
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
#     model.decoder = get_peft_model(model.decoder, lora_config)
#     model.decoder.gradient_checkpointing_enable()

#     # 7) (Optional) manually create device_map
#     all_submodules = list(model.named_modules())
#     layer_submodules = []
#     for name, _ in all_submodules:
#         match = re.match(r"decoder\.base_model\.model\.model\.layers\.(\d+)(\..*)?$", name)
#         if match:
#             idx = int(match.group(1))
#             layer_submodules.append((idx, name))

#     layer_submodules.sort(key=lambda x: x[0])
#     half = len(layer_submodules) // 2

#     device_map = {}
#     # for i, (layer_index, layer_name) in enumerate(layer_submodules):
#     #     if i < half:
#     #         device_map[layer_name] = 0
#     #     else:
#     #         device_map[layer_name] = 1

#     # # If you want the GCN on GPU 1
#     # device_map["gcn"] = 1
#     # # The root module on GPU 0
#     # device_map[""] = 0

#     # print("device_map:", device_map)
#     model = dispatch_model(model, device_map=device_map)

#     # 8) Prepare data + tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name,
#         padding_side='left',
#         add_eos_token=True,
#         trust_remote_code=True
#     )
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     training_batch = utils.prepare_batch(
#         tokenizer,
#         all_result,
#         args.num_train_samples,
#         args.max_node_num
#     )

#     graphs = utils.construct_graphs(
#         all_result,
#         node_embeddings,
#         args.num_train_samples,
#         args.max_node_num
#     )
#     training_batch, graphs = utils.pad_graphs(training_batch, graphs)

#     dataset = list(zip(training_batch, graphs))

#     from torch.utils.data import DataLoader
#     def collate_fn(samples):
#         batch_list, graph_list = [], []
#         for s in samples:
#             batch_list.append(s[0])
#             graph_list.append(s[1])
#         return batch_list, graph_list

#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         collate_fn=lambda x: (x, x)
#     )

#     # 9) Optimizer
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

#     # 10) Train loop
#     for epoch in range(args.num_epochs):
#         avg_loss = train_one_epoch(model, dataloader, optimizer)
#         print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f}")

#     # 11) Save LoRA
#     model.decoder.save_pretrained("MAGDi_SQA_lora_adapters_8bit")
#     print("Training complete and saved to MAGDi_SQA_lora_adapters_8bit")

