# train.py
import os
import json
import torch
import torch.nn as nn
import pickle
import numpy as np
import argparse
import re
from collections import defaultdict

np.random.seed(0)

import deepspeed
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from accelerate import dispatch_model

from model import MAGDi  # your custom class
import utils            # your utility file
import data_utils       # your data collator or loading utilities
import networkx as nx

torch.cuda.empty_cache()

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0

    for step, batch_data in enumerate(dataloader):
        batch, graph = batch_data
        
        # Custom forward:
        nll_loss, node_loss, mr_loss = model(
            pos_input_ids=batch["pos_input_ids"],
            pos_attention_mask=batch["pos_attention_mask"],
            pos_labels=batch["pos_labels"],
            neg_input_ids=batch["neg_input_ids"],
            neg_attention_mask=batch["neg_attention_mask"],
            neg_labels=batch["neg_labels"],
            graph=graph
        )
        loss = nll_loss + node_loss + mr_loss

        # DeepSpeed handles backward and optimizer.step()
        model.backward(loss)
        model.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SQA', type=str)
    parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-Coder-V2-Lite-Base', type=str)
    parser.add_argument('--gcn_in_channels', default=2048, type=int)
    parser.add_argument('--gcn_hidden_channels', default=512, type=int)
    parser.add_argument('--gcn_out_channels', default=3, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--num_train_samples', default=1000, type=int)
    parser.add_argument('--max_node_num', default=12, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    
    # Add QLoRA parameters
    parser.add_argument('--bits', type=int, default=8,
                        help='Quantization bits (4 or 8)')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # Add DeepSpeed arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json',
                       help='DeepSpeed configuration file')
    
    # Parse args
    args = parser.parse_args()

    
    # 1) Load data + embeddings
    with open(f"node_emb/{args.dataset}_node_emb.pkl", "rb") as f:
        node_embeddings = pickle.load(f)

    with open(f"MAG/{args.dataset}_1000.json", "r") as f:
        all_result = json.load(f)
    all_result = all_result[:args.num_train_samples]

    node_embeddings = node_embeddings.reshape(
        args.num_train_samples,
        args.max_node_num,
        -1
    )
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float32)

    # 2) QLoRA: Configure BitsAndBytes 8-bit quantization
    compute_dtype = torch.float32

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,  # Double quantization for better memory efficiency
        bnb_4bit_quant_type="nf4",       # Normalized Float 4 for better accuracy
    )

    # 3) Load base model in 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto"
    )

    # 4) Prepare for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # 5) Build MAGDi with quantized base
    model = MAGDi(
        base_model=base_model,
        gcn_in_channels=args.gcn_in_channels,
        gcn_hidden_channels=args.gcn_hidden_channels,
        gcn_out_channels=args.gcn_out_channels,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )

    # Freeze base model weights
    for param in model.decoder.parameters():
        param.requires_grad = False

    # 6) QLoRA configuration
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "deepseek" in args.model_name.lower():
        # For DeepSeek models
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model.decoder = get_peft_model(model.decoder, lora_config)
    model.decoder.gradient_checkpointing_enable()

    # 7) Prepare data + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side='left',
        add_eos_token=True,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    training_batch = utils.prepare_batch(
        tokenizer,
        all_result,
        args.num_train_samples,
        args.max_node_num
    )

    graphs = utils.construct_graphs(
        all_result,
        node_embeddings,
        args.num_train_samples,
        args.max_node_num
    )
    training_batch, graphs = utils.pad_graphs(training_batch, graphs)

    dataset = list(zip(training_batch, graphs))

    # Custom collate function to properly handle batching
    def collate_fn(samples):
        batch_list, graph_list = [], []
        for s in samples:
            batch_list.append(s[0])
            graph_list.append(s[1])
        
        # Process batch_list to combine dictionary values
        combined_batch = defaultdict(list)
        for batch in batch_list:
            for k, v in batch.items():
                combined_batch[k].append(v)
        
        # Stack tensors or concatenate as appropriate
        for k in combined_batch:
            if isinstance(combined_batch[k][0], torch.Tensor):
                combined_batch[k] = torch.stack(combined_batch[k])
        
        return combined_batch, graph_list

    # Create DataLoader with distributed sampler
    from torch.utils.data import DataLoader, DistributedSampler
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )

    # 8) Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=args.deepspeed_config
    )

    # 9) Train loop
    for epoch in range(args.num_epochs):
        # Set dataloader sampler's epoch for deterministic shuffling
        dataloader.sampler.set_epoch(epoch)
            
        avg_loss = train_one_epoch(model_engine, dataloader, optimizer)
        
        # Only print from rank 0
        if torch.distributed.get_rank() == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f}")

    # After line 260, add:
    if torch.distributed.get_rank() == 0:
        # Save LoRA adapters
        output_dir = f"MAGDi_{args.dataset}_qlora_adapters"
        model_engine.module.decoder.save_pretrained(output_dir)
        
        # Also save quantization and model configuration
        model_engine.module.decoder.config.save_pretrained(output_dir)
        
        # Merge adapters into the base model
        print("Merging LoRA adapters into base model...")
        merged_model_path = f"MAGDi_{args.dataset}_merged_model"
        
        # Get the base model architecture without quantization
        base_model_unquantized = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        
        # Load the trained adapters
        from peft import PeftModel
        merged_model = PeftModel.from_pretrained(base_model_unquantized, output_dir)
        
        # Merge weights
        merged_model = merged_model.merge_and_unload()
        
        # Save the full model
        merged_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        
        print(f"Training complete. Adapters saved to {output_dir} and merged model saved to {merged_model_path}")