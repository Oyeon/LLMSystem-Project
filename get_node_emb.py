import json
import torch
import pickle
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

import utils  # must contain generate_ordered_list(...) for your SQA data

def generate_node_embeddings(
    input_json="MAG/SQA_1000.json",
    output_pkl="node_emb/SQA_node_emb.pkl",
    model_name="deepseek-ai/DeepSeek-V2-Lite",
    batch_size=1
):
    """
    Generates node embeddings from the final hidden layer of the DeepSeek model
    using manual memory management for large models.
    """
    # 1) Load data
    with open(input_json, "r") as f:
        sqa_data = json.load(f)

    # 2) Convert data to a list of strings/nodes
    ordered_list, labels = utils.generate_ordered_list(sqa_data)
    print(f"Number of items in 'ordered_list': {len(ordered_list)}")

    # 3) Load the DeepSeek model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=True,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use more aggressive CPU offloading for large models
    # Process the model layer by layer instead of all at once
    print("Loading model with CPU offloading...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        max_memory={0: "23GB", "cpu": "10GB"}  # Limit CPU memory usage
    )
    
    model.eval()

    # Get hidden size from config
    print("Model hidden_size from config:", model.config.hidden_size)
    hidden_size = model.config.hidden_size
    
    # 4) Generate embeddings by mean-pooling the last hidden state
    node_embeddings = None

    for i in tqdm(range(0, len(ordered_list), batch_size)):
        batch_texts = ordered_list[i : i + batch_size]
        
        # Tokenize
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Reduced max length to avoid OOM
        )
        
        # Process in smaller chunks if needed
        with torch.no_grad():
            # Move inputs to device as needed by the model
            tokens = {k: v.to(model.device) for k, v in tokens.items()}
            
            # Run with memory optimization flags
            outputs = model(
                **tokens,
                output_hidden_states=True,
                return_dict=True
            )

        # Get the final hidden layer
        last_hidden_state = outputs.hidden_states[-1]

        # Move to CPU immediately to free GPU memory
        last_hidden_state = last_hidden_state.cpu()
        
        # Weighted approach for ignoring padding (on CPU)
        seq_len = last_hidden_state.shape[1]
        weights_for_non_padding = (
            tokens['attention_mask'].cpu()
            * torch.arange(1, seq_len + 1).unsqueeze(0)
        )
        sum_node_embeddings = torch.sum(
            last_hidden_state * weights_for_non_padding.unsqueeze(-1),
            dim=1
        )
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        
        # Compute final embeddings on CPU
        emb_batch = (sum_node_embeddings / num_of_none_padding_tokens).numpy()

        if node_embeddings is None:
            node_embeddings = emb_batch
        else:
            node_embeddings = np.concatenate([node_embeddings, emb_batch], axis=0)
            
        # Explicitly clean up memory
        del outputs, last_hidden_state, sum_node_embeddings, num_of_none_padding_tokens
        torch.cuda.empty_cache()

    print(f"Final shape of node_embeddings: {node_embeddings.shape}")

    # 5) Save to .pkl
    with open(output_pkl, "wb") as f:
        pickle.dump(node_embeddings, f)
    print(f"Saved new embeddings to {output_pkl}")


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    
    # Add custom arguments
    parser.add_argument('--input_json', type=str, default="MAG/SQA_1000.json")
    parser.add_argument('--output_pkl', type=str, default="node_emb/SQA_node_emb.pkl")
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-V2-Lite")
    parser.add_argument('--batch_size', type=int, default=1)
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)
    # Create offload folder if it doesn't exist
    os.makedirs("offload_folder", exist_ok=True)
    
    generate_node_embeddings(
        input_json=args.input_json,
        output_pkl=args.output_pkl,
        model_name=args.model_name,
        batch_size=args.batch_size
    )