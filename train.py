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

    # single "main" device if model is split across multiple GPUs
    main_device = next(model.parameters()).device

    for step, (batch, graph) in enumerate(dataloader):
        # Move batch + graph to main_device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(main_device)
        if hasattr(graph, 'to'):
            graph = graph.to(main_device)

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

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SQA', type=str)
    parser.add_argument('--model_name', default='mistralai/Mistral-7B-Instruct-v0.2', type=str)
    parser.add_argument('--gcn_in_channels', default=4096, type=int)
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

    # 2) BitsAndBytesConfig for 8-bit
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_memory_efficient_backward=False  # <--- crucial
    )

    # 3) Load base model in 8-bit
    base_model_8bit = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,  # not load_in_8bit=...
        device_map="auto"  # or "cpu"
    )

    # 4) Prepare for k-bit training
    base_model_8bit = prepare_model_for_kbit_training(base_model_8bit)

    # 5) Build MAGDi with that 8-bit base
    model = MAGDi(
        base_model=base_model_8bit,
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

    # 6) LoRA config + wrap
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.decoder = get_peft_model(model.decoder, lora_config)
    model.decoder.gradient_checkpointing_enable()

    # 7) (Optional) manually create device_map
    all_submodules = list(model.named_modules())
    layer_submodules = []
    for name, _ in all_submodules:
        match = re.match(r"decoder\.base_model\.model\.model\.layers\.(\d+)(\..*)?$", name)
        if match:
            idx = int(match.group(1))
            layer_submodules.append((idx, name))

    layer_submodules.sort(key=lambda x: x[0])
    half = len(layer_submodules) // 2

    device_map = {}
    # for i, (layer_index, layer_name) in enumerate(layer_submodules):
    #     if i < half:
    #         device_map[layer_name] = 0
    #     else:
    #         device_map[layer_name] = 1

    # # If you want the GCN on GPU 1
    # device_map["gcn"] = 1
    # # The root module on GPU 0
    # device_map[""] = 0

    # print("device_map:", device_map)
    model = dispatch_model(model, device_map=device_map)

    # 8) Prepare data + tokenizer
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

    from torch.utils.data import DataLoader
    def collate_fn(samples):
        batch_list, graph_list = [], []
        for s in samples:
            batch_list.append(s[0])
            graph_list.append(s[1])
        return batch_list, graph_list

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: (x, x)
    )

    # 9) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 10) Train loop
    for epoch in range(args.num_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f}")

    # 11) Save LoRA
    model.decoder.save_pretrained("MAGDi_SQA_lora_adapters_8bit")
    print("Training complete and saved to MAGDi_SQA_lora_adapters_8bit")


### ? 1st 
# import os
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
# import torch
# torch.cuda.empty_cache()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # dataset can be: ['SQA', 'ECQA', 'ARC', 'GSM8K', 'MATH']
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
#     args = parser.parse_args()

#     # 1) Load the new 2048-dim embeddings
#     with open(f"node_emb/{args.dataset}_node_emb.pkl", "rb") as f:
#         node_embeddings = pickle.load(f)

#     # 2) Load your data
#     with open(f"MAG/{args.dataset}_1000.json", "r") as f:
#         all_result = json.load(f)
#     all_result = all_result[:args.num_train_samples]

#     # 3) Initialize the MAGDi model
#     model = MAGDi(
#         model_name=args.model_name,
#         gcn_in_channels=args.gcn_in_channels,
#         gcn_hidden_channels=args.gcn_hidden_channels,
#         gcn_out_channels=args.gcn_out_channels,
#         alpha=args.alpha,
#         beta=args.beta,
#         gamma=args.gamma
#     )

#     # Check hidden size is 2048
#     print("Loaded model hidden_size =", model.decoder.config.hidden_size)
#     assert model.decoder.config.hidden_size == 2048, "Double-check the model is 2048!"

#     # 4) Reshape node_embeddings
#     node_embeddings = node_embeddings.reshape(
#         args.num_train_samples,
#         args.max_node_num,
#         model.decoder.config.hidden_size
#     )
#     node_embeddings = torch.tensor(node_embeddings)
#     print("Final node_embeddings shape:", node_embeddings.size())  # [1000, 12, 2048]

#     # 5) Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name,
#         padding_side='left',
#         add_eos_token=True,
#         trust_remote_code=True
#     )
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     # 6) Set up device_map to force everything to GPU:1
#     # Option A: Entire model on GPU:1
#     # device_map = {"": 1}
#     # model = dispatch_model(model, device_map=device_map)

#     max_memory = {
#         0: "0GiB",     # If you want to avoid GPU 0
#         1: "48GiB",    # e.g. total memory, or smaller if you want to be conservative
#         "cpu": "96GiB"
#     }

#     device_map = infer_auto_device_map(
#         model,
#         max_memory=max_memory,
#         no_split_module_classes=["GCN"],  # or whatever you want
#         dtype='float16'
#     )

#     model = dispatch_model(model, device_map=device_map)

#     # 7) Freeze base model weights (LoRA approach)
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

#     # If needed, cast final lm_head to float32
#     model.decoder.lm_head = utils.CastOutputToFloat(model.decoder.lm_head)

#     # 8) Prepare data batches
#     training_batch = utils.prepare_batch(
#         tokenizer,
#         all_result,
#         args.num_train_samples,
#         args.max_node_num
#     )

#     # 9) Construct graphs and pad them
#     graphs = utils.construct_graphs(
#         all_result,
#         node_embeddings,
#         args.num_train_samples,
#         args.max_node_num
#     )
#     training_batch, graphs = utils.pad_graphs(training_batch, graphs)
#     print(f"Length training_batch: {len(training_batch)}, length graphs: {len(graphs)}")

#     # 10) Set up the HuggingFace Trainer
#     trainer = MAGDiTrainer(
#         model=model,
#         train_dataset=training_batch,
#         args=transformers.TrainingArguments(
#             per_device_train_batch_size=1,
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

#     # 11) Train
#     trainer.train()

#     # 12) Save the LoRA adapters
#     model.decoder.save_pretrained("MAGDi_SQA")
#     print("Training complete and saved to MAGDi_SQA")


##### ORIGINAL
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



# custom_dir = './nas_ssd2/'

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # dataset: ['SQA', 'ECQA', 'ARC', 'GSM8K', 'MATH']
#     parser.add_argument('--dataset', default='SQA', type=str)
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

#     node_embeddings = node_embeddings.reshape(args.num_train_samples, args.max_node_num, model.decoder.config.hidden_size)
#     node_embeddings = torch.tensor(node_embeddings)
#     node_embeddings = node_embeddings[:args.num_train_samples, :, :]
#     node_embeddings.size()

#     tokenizer = AutoTokenizer.from_pretrained(args.model_name,
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

#     device_map = {"": 0}  # Put every submodule on GPU 1

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
