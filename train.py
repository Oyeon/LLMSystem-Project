# train_hf.py
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

# Import HuggingFace Trainer
from transformers import Trainer, TrainingArguments

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

from transformers.integrations import HfDeepSpeedConfig
from accelerate import dispatch_model

from model import MAGDi  # your custom class
import utils            # your utility file
import data_utils       # your data collator or loading utilities
import networkx as nx

# Add this before loading the model
import os
os.environ['TRITON_CACHE_DIR'] = '/ocean/projects/cis240137p/adas11/triton_cache'
os.environ['BITSANDBYTES_CEXTENSION_PATH'] = '/ocean/projects/cis240137p/adas11/bnb_cache'
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = '/ocean/projects/cis240137p/adas11/pytorch_kernel_cache'
os.environ['TORCH_EXTENSIONS_DIR'] = '/ocean/projects/cis240137p/adas11/project/torch_extensions'
os.environ['HF_HOME'] = "/ocean/projects/cis240137p/adas11/hf_cache"

torch.cuda.empty_cache()

# Create a custom dataset class
class MAGDataset(torch.utils.data.Dataset):
    def __init__(self, batch_data, graphs):
        self.batch_data = batch_data
        self.graphs = graphs
    
    def __len__(self):
        return len(self.batch_data)
    
    def __getitem__(self, idx):
        item = self.batch_data[idx]
        graph = self.graphs[idx]
        
        # Convert lists to tensors if needed
        for key, value in item.items():
            if isinstance(value, list):
                item[key] = torch.tensor(value, dtype=torch.long)
        
        return {
            "pos_input_ids": item["pos_input_ids"],
            "pos_attention_mask": item["pos_attention_mask"], 
            "pos_labels": item["pos_labels"],
            "neg_input_ids": item["neg_input_ids"],
            "neg_attention_mask": item["neg_attention_mask"],
            "neg_labels": item["neg_labels"],
            "graph": graph
        }

# Create a custom data collator
class MAGDataCollator:
    def __call__(self, features):
        batch = {}
        graph_batch = []
        
        for feature in features:
            graph_batch.append(feature.pop("graph"))
        
        # Process the remaining features into a batch
        for key in features[0].keys():
            values = [feature[key] for feature in features]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values
        
        batch["graph"] = graph_batch
        return batch

# Create a custom trainer
class MAGDiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        nll_loss, node_loss, mr_loss = model(
            pos_input_ids=inputs["pos_input_ids"],
            pos_attention_mask=inputs["pos_attention_mask"],
            pos_labels=inputs["pos_labels"],
            neg_input_ids=inputs["neg_input_ids"],
            neg_attention_mask=inputs["neg_attention_mask"],
            neg_labels=inputs["neg_labels"],
            graph=inputs["graph"]
        )
        
        loss = nll_loss + node_loss + mr_loss
        
        if return_outputs:
            return loss, (nll_loss, node_loss, mr_loss)
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SQA', type=str)
    parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-V2-Lite', type=str)
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
    parser.add_argument('--bits', type=int, default=4,
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
    compute_dtype = torch.float16
    node_embeddings = torch.tensor(node_embeddings, dtype=compute_dtype)
    
    # Initialize distributed training if needed
    if args.local_rank != -1:
        deepspeed.init_distributed()
        torch.cuda.set_device(args.local_rank)
        print(f"Process rank: {torch.distributed.get_rank()}, using GPU: {args.local_rank}")
    
    dschf = HfDeepSpeedConfig(args.deepspeed_config)  # keep this object alive

    # 2) QLoRA: Configure BitsAndBytes 8-bit quantization


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,  # Double quantization for better memory efficiency
        bnb_4bit_quant_type="nf4",       # Normalized Float 4 for better accuracy
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        device_map="cuda"  # Place on first GPU
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
#    model.decoder.gradient_checkpointing_enable()

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

    print("Sample batch item:", training_batch[0])

    graphs = utils.construct_graphs(
        all_result,
        node_embeddings,
        args.num_train_samples,
        args.max_node_num
    )
    training_batch, graphs = utils.pad_graphs(training_batch, graphs)

    # Create dataset
    dataset = MAGDataset(training_batch, graphs)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=f"MAGDi_{args.dataset}_checkpoints",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        # deepspeed=args.deepspeed_config,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        local_rank=args.local_rank,
        fp16=True,
        save_total_limit=2,
        remove_unused_columns=False,  # Important for custom datasets
    )

    # Create trainer
    trainer = MAGDiTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MAGDataCollator(),
    )

    # Train
    trainer.train()

    # Save model
    if trainer.is_world_process_zero():
        # Save LoRA adapters
        output_dir = f"MAGDi_{args.dataset}_qlora_adapters"
        trainer.model.decoder.save_pretrained(output_dir)
        
        # Also save quantization and model configuration
        trainer.model.decoder.config.save_pretrained(output_dir)
        
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
