import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import utils  # must contain generate_ordered_list(...) for your SQA data

def generate_node_embeddings(
    input_json="MAG/SQA_1000.json",
    output_pkl="node_emb/SQA_node_emb.pkl",
    model_name='mistralai/Mistral-7B-Instruct-v0.2',
    batch_size=50
):
    """
    Generates 2048-dim node embeddings from the final hidden layer of the DeepSeek model
    and saves them to a .pkl file. We confirm the hidden size is 2048 by printing/asserting.
    """
    # 1) Load data
    with open(input_json, "r") as f:
        sqa_data = json.load(f)

    # 2) Convert data to a list of strings/nodes
    #    e.g., "ordered_list" has 1000 samples * 12 nodes = 12000 items
    ordered_list, labels = utils.generate_ordered_list(sqa_data)
    print(f"Number of items in 'ordered_list': {len(ordered_list)}")

    # 3) Load the DeepSeek model (2048 hidden size)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=True,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # or "cuda:0"
        trust_remote_code=True
    )
    model.eval()

    # Confirm hidden size is 2048:
    print("Model hidden_size from config:", model.config.hidden_size)
    assert model.config.hidden_size == 4096, (
        "It appears the model does not have a 2048 hidden size. "
        "If it's actually 4096, you need to change your GCN config and reshape logic accordingly."
    )

    # 4) Generate embeddings by mean-pooling the last hidden state
    node_embeddings = None

    for i in tqdm(range(0, len(ordered_list), batch_size)):
        batch_texts = ordered_list[i : i + batch_size]
        
        # Tokenize
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        # Get the final hidden layer: [batch_size, seq_len, 2048]
        last_hidden_state = outputs.hidden_states[-1]

        # Weighted approach for ignoring padding
        # We'll multiply each token embedding by an increasing index
        # so that trailing 0 pads don't incorrectly accumulate
        seq_len = last_hidden_state.shape[1]
        weights_for_non_padding = (
            tokens.attention_mask
            * torch.arange(1, seq_len + 1, device=model.device).unsqueeze(0)
        )
        sum_node_embeddings = torch.sum(
            last_hidden_state * weights_for_non_padding.unsqueeze(-1),
            dim=1
        )
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        emb_batch = (sum_node_embeddings / num_of_none_padding_tokens).cpu().numpy()  # [batch_size, 2048]

        if node_embeddings is None:
            node_embeddings = emb_batch
        else:
            node_embeddings = np.concatenate([node_embeddings, emb_batch], axis=0)

    # Now shape should be [N, 2048], where N = len(ordered_list)
    print("Final shape of node_embeddings:", node_embeddings.shape)

    # 5) Save to .pkl
    with open(output_pkl, "wb") as f:
        pickle.dump(node_embeddings, f)
    print(f"Saved new embeddings to {output_pkl}")


if __name__ == "__main__":
    generate_node_embeddings(
        input_json="MAG/SQA_1000.json",         # your SQA data
        output_pkl="node_emb/SQA_node_emb.pkl", # store new 2048-dim embeddings
        model_name='mistralai/Mistral-7B-Instruct-v0.2',
        batch_size=2
    )


# import json
# import torch
# import pickle
# import numpy as np
# from tqdm.notebook import tqdm
# from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM

# with open("MAG_ARC.json", "r") as f:
#     MAGs = json.load(f)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     device_map="auto")

# tokenizer = AutoTokenizer.from_pretrained(model_name,
#                                           padding_side='left',
#                                           add_eos_token=True)

# tokenizer.pad_token_id = tokenizer.eos_token_id
# ordered_list, labels = utils.generate_ordered_list(MAGs)

# node_embeddings = None
# batch_size = 50

# for i in tqdm(range(0, len(ordered_list), batch_size)):
#     batch = ordered_list[i: i+batch_size]
#     tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
#     with torch.no_grad():
#         outputs = model(**tokens, output_hidden_states=True)
#     last_hidden_state = outputs.hidden_states[-1]
#     weights_for_non_padding = tokens.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).to(tokens.attention_mask.device).unsqueeze(0)
#     weights_for_non_padding = weights_for_non_padding.to(last_hidden_state.device)
#     sum_node_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
#     num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
#     emb = (sum_node_embeddings / num_of_none_padding_tokens).detach().cpu().numpy()
        
#     if node_embeddings is None:
#         node_embeddings = emb
#     else:
#         node_embeddings = np.concatenate([node_embeddings, emb])
# print(node_embeddings.shape)

# with open("ARC_node_emb.pkl", "wb") as f:
#     pickle.dump(node_embeddings, f)