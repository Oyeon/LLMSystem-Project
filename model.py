# model.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Linear
from torch.nn import CrossEntropyLoss, MarginRankingLoss

# --- GCN Class (remains the same) ---
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, dtype=torch.float16):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
        self.dtype = dtype

    # Consider the refined forward from previous suggestions for stability
    def forward(self, x, edge_index):
        # Ensure input is in the desired model dtype
        x = x.to(self.dtype)
        original_dtype = self.dtype

        # Explicitly use float32 for GCNConv computation (safer)
        x_compute = x.to(torch.float32)
        h = self.gcn1(x_compute, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index) # h is float32 here

        # Convert GCN embedding output back to original/target dtype
        gcn_embeddings_out = h.to(original_dtype)

        # Calculate log_softmax - usually safer in float32 for numerical stability
        # CrossEntropyLoss expects raw logits (not log_softmax)
        # Return raw logits in float32
        return gcn_embeddings_out, h # Return embeddings and float32 logits
# --- End GCN Class ---


class MAGDi(torch.nn.Module):
    """
    Refactored to perform only one forward pass through the decoder.
    """
    def __init__(
        self,
        base_model,
        gcn_in_channels,
        gcn_hidden_channels,
        gcn_out_channels,
        alpha,
        beta,
        gamma,
        torch_dtype=torch.float16,
        label_ignore_index=-100 # Standard ignore index for CrossEntropyLoss
    ):
        super().__init__()
        self.decoder = base_model
        self.dtype = torch_dtype
        self.label_ignore_index = label_ignore_index

        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels, dtype=self.dtype)

        # Ensure MLPs handle the correct dtype
        self.mlp1 = Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size).to(self.dtype)
        self.mlp2 = Linear(self.decoder.config.hidden_size, 1).to(self.dtype)
        # self.mlp3 = Linear(self.decoder.config.vocab_size, 1) # This seemed unused previously

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        pos_input_ids,
        pos_attention_mask,
        pos_labels,
        neg_input_ids,
        neg_attention_mask,
        neg_labels, # Note: neg_labels are not used for NLL loss calculation now
        graph
    ):
        # Assuming inputs['graph'] is a list of graph objects from the collator
        # Determine batch size BEFORE concatenation
        original_batch_size = pos_input_ids.size(0)
        device = pos_input_ids.device # Use device from a tensor guaranteed to be on target device

        # --- 1) Prepare Graph Batch ---
        # Ensure graph data is on the correct device
        # Note: DataLoader might be inefficient here if graph is already a list
        # If 'graph' is already a list of Data objects, batch them differently if needed
        # Or pass the pre-batched graph if possible. Assuming current graph handling is desired.
        try:
            loader = DataLoader(graph, batch_size=original_batch_size, shuffle=False, pin_memory=False, num_workers=0)
            graph_batch = next(iter(loader)).to(device)
        except Exception as e:
            print(f"Error creating DataLoader for graph: {e}")
            # Handle error or alternative graph batching
            return torch.tensor(0.0, requires_grad=True, device=device), \
                   torch.tensor(0.0, requires_grad=True, device=device), \
                   torch.tensor(0.0, requires_grad=True, device=device)


        # --- 2) Combine Inputs for Single Decoder Pass ---
        # Ensure consistent sequence lengths (assuming padding is handled by data prep)
        # If not, padding would be required here before concatenation.
        combined_input_ids = torch.cat([pos_input_ids, neg_input_ids], dim=0)
        combined_attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)

        # Create combined labels: pos_labels for positive, ignore_index for negative
        neg_labels_ignore = torch.full_like(
            neg_input_ids, self.label_ignore_index
        )
        combined_labels = torch.cat([pos_labels, neg_labels_ignore], dim=0)

        # --- 3) Single Decoder Forward Pass ---
        # The 'loss' returned here will be the NLL loss calculated only on the positive examples
        # because the negative examples have labels set to the ignore index.
        combined_output = self.decoder(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            output_hidden_states=True,
            return_dict=True # Ensure output is a dictionary
        )

        # Extract NLL loss (calculated only on positive samples)
        nll_loss = combined_output.loss

        # Extract combined hidden states (shape: [2 * B, SeqLen, HiddenDim])
        combined_hidden_states = combined_output.hidden_states[-1]

        # --- 4) Split Hidden States and Masks ---
        # Split based on the original batch size
        pos_hidden_states = combined_hidden_states[:original_batch_size]
        neg_hidden_states = combined_hidden_states[original_batch_size:]

        # We need the original masks for correct pooling
        # pos_attention_mask_pool = combined_attention_mask[:original_batch_size] # This is just original pos_attention_mask
        # neg_attention_mask_pool = combined_attention_mask[original_batch_size:] # This is just original neg_attention_mask

        # --- 5) Mean Pool Hidden States ---
        pos_seq_emb = self._mean_pool(pos_hidden_states, pos_attention_mask)
        neg_seq_emb = self._mean_pool(neg_hidden_states, neg_attention_mask)

        # --- 6) Filter Short Negative Sequences (Optional, based on original logic) ---
        # Use the *original* neg_attention_mask to determine length
        row_sums = neg_attention_mask.sum(dim=1)
        neg_mask = row_sums > 5 # Mask for valid negative examples

        # Apply mask *after* pooling if needed
        if neg_mask.any() and neg_mask.size(0) == original_batch_size:
            # Ensure the mask is applied consistently if filtering happens
            # If we filter, we must filter both pos and neg embeddings/scores
            # to keep pairs aligned for MarginRankingLoss.
            valid_indices = neg_mask.to(device) # Convert mask to boolean tensor on correct device
            pos_seq_emb = pos_seq_emb[valid_indices]
            neg_seq_emb = neg_seq_emb[valid_indices]
            # Check if any samples remain after filtering
            if pos_seq_emb.numel() == 0 or neg_seq_emb.numel() == 0:
                 # Handle case where filtering removed all samples
                 print("Warning: Filtering removed all samples for MarginRankingLoss.")
                 mr_loss = torch.tensor(0.0, device=device, requires_grad=True) # Assign zero loss
            else:
                 # Proceed with MLP and MarginRankingLoss only if samples remain
                 pos_h = torch.relu(self.mlp1(pos_seq_emb.to(self.dtype))) # Ensure dtype
                 pos_score = torch.tanh(self.mlp2(pos_h))

                 neg_h = torch.relu(self.mlp1(neg_seq_emb.to(self.dtype))) # Ensure dtype
                 neg_score = torch.tanh(self.mlp2(neg_h))

                 # --- 7) MarginRankingLoss ---
                 mr_cri = MarginRankingLoss(margin=1.0, reduction='mean').to(device)
                 mr_loss = mr_cri(pos_score, neg_score, torch.ones_like(pos_score).to(device))
        elif original_batch_size > 0: # Calculate MR loss if no filtering or all were valid
            pos_h = torch.relu(self.mlp1(pos_seq_emb.to(self.dtype))) # Ensure dtype
            pos_score = torch.tanh(self.mlp2(pos_h))

            neg_h = torch.relu(self.mlp1(neg_seq_emb.to(self.dtype))) # Ensure dtype
            neg_score = torch.tanh(self.mlp2(neg_h))

            # --- 7) MarginRankingLoss ---
            mr_cri = MarginRankingLoss(margin=1.0, reduction='mean').to(device)
            mr_loss = mr_cri(pos_score, neg_score, torch.ones_like(pos_score).to(device))
        else: # Handle batch size 0 case
            mr_loss = torch.tensor(0.0, device=device, requires_grad=True)


        # --- 8) GCN Forward Pass ---
        # Ensure graph features have the correct dtype for GCN
        gcn_node_features = graph_batch.x.to(self.gcn.dtype)
        # Make sure edge_index is LongTensor
        gcn_edge_index = graph_batch.edge_index.to(torch.long)

        gcn_out_embeddings, gcn_logits = self.gcn(gcn_node_features, gcn_edge_index)
        # Ensure target labels are LongTensor and on the correct device
        gcn_targets = graph_batch.y.to(device=device, dtype=torch.long)

        # Use CrossEntropyLoss with raw logits (float32 recommended from GCN)
        node_loss_cri = CrossEntropyLoss().to(device)
        node_loss = node_loss_cri(gcn_logits, gcn_targets)


        # --- 9) Return Weighted Losses ---
        # Handle potential NaN/Inf losses gracefully
        if torch.isnan(nll_loss) or torch.isinf(nll_loss):
             print("Warning: NaN or Inf detected in nll_loss. Setting to 0.")
             nll_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if torch.isnan(node_loss) or torch.isinf(node_loss):
             print("Warning: NaN or Inf detected in node_loss. Setting to 0.")
             node_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if torch.isnan(mr_loss) or torch.isinf(mr_loss):
             print("Warning: NaN or Inf detected in mr_loss. Setting to 0.")
             mr_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return (
            self.alpha * nll_loss,
            self.beta * node_loss,
            self.gamma * mr_loss
        )

    # --- _mean_pool method (remains the same) ---
    def _mean_pool(self, hidden_states, attention_mask):
        # Ensure hidden_states are float for calculations if needed
        hidden_states = hidden_states.float()
        weights = attention_mask.float() * torch.arange(
            1, hidden_states.shape[1] + 1, device=hidden_states.device, dtype=torch.float
        ).unsqueeze(0)
        sum_embeddings = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        denom = torch.sum(weights, dim=1).unsqueeze(-1)
        # Add epsilon to avoid division by zero if attention mask is all zeros
        denom = denom + 1e-8
        pooled = sum_embeddings / denom
        # Return in the original or desired dtype
        return pooled.to(self.dtype)