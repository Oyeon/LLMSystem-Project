# model.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Linear
from torch.nn import CrossEntropyLoss, MarginRankingLoss

# Import our fused kernel module
from fused_tanh_margin import FusedTanhMarginloss

class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
    
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gcn2(x, edge_index)
        return x, F.log_softmax(x, dim=1)

class MAGDi(torch.nn.Module):
    """
    Updated to accept `base_model` as a parameter:
    it should be an AutoModelForCausalLM that was loaded in 8-bit.
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
        use_fused_kernel=True  # New parameter to enable/disable the fused kernel
    ):
        super().__init__()
        self.decoder = base_model  # The 8-bit base model
        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)

        self.mlp1 = Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.mlp2 = Linear(self.decoder.config.hidden_size, 1)
        self.mlp3 = Linear(self.decoder.config.vocab_size, 1)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize our fused tanh + margin ranking loss module
        self.fused_tanh_marginloss = FusedTanhMarginloss(margin=1.0, use_fused_kernel=use_fused_kernel)
        
    def forward(
        self,
        pos_input_ids,
        pos_attention_mask,
        pos_labels,
        neg_input_ids,
        neg_attention_mask,
        neg_labels,
        graph
    ):

        device = pos_input_ids.device

        # 1) Graph
        loader = DataLoader(graph, batch_size=len(graph), shuffle=False, pin_memory=False, num_workers=0)
        graph_batch = next(iter(loader)).to(device)

        # 2) Positive
        pos_output = self.decoder(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            labels=pos_labels,
            output_hidden_states=True
        )
        nll_loss = pos_output["loss"]

        # 3) Negative
        neg_output = self.decoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            labels=None,
            output_hidden_states=True
        )

        # Possibly mask out short negative sequences
        row_sums = neg_attention_mask.sum(dim=1)
        neg_mask = row_sums > 5
        
        # 4) Mean pool hidden states
        pos_seq_emb = self._mean_pool(pos_output.hidden_states[-1], pos_attention_mask)
        neg_seq_emb = self._mean_pool(neg_output.hidden_states[-1], neg_attention_mask)

        if neg_mask.any():
            neg_mask = neg_mask.to(device)
            pos_seq_emb = pos_seq_emb[neg_mask]
            neg_seq_emb = neg_seq_emb[neg_mask]
            
        # MLP
        pos_h = torch.relu(self.mlp1(pos_seq_emb))
        neg_h = torch.relu(self.mlp1(neg_seq_emb))
        
        # Replace the separate tanh and MarginRankingLoss with our fused implementation
        pos_input = self.mlp2(pos_h)
        neg_input = self.mlp2(neg_h)
        
        mr_loss, pos_score, neg_score = self.fused_tanh_marginloss(pos_input, neg_input)

        # 6) GCN
        gcn_out, logits = self.gcn(graph_batch.x, graph_batch.edge_index)
        graph_batch.y = graph_batch.y.to(device)
        node_loss = CrossEntropyLoss()(logits, graph_batch.y)

        return (
            self.alpha * nll_loss,
            self.beta * node_loss,
            self.gamma * mr_loss
        )

    def _mean_pool(self, hidden_states, attention_mask):
        # Weighted approach as in your code:
        weights = attention_mask * torch.arange(
            1, hidden_states.shape[1] + 1, device=hidden_states.device
        ).unsqueeze(0)
        sum_embeddings = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        denom = torch.sum(weights, dim=1).unsqueeze(-1)
        return sum_embeddings / denom

# import torch
# import logging
# import torch.nn.functional as F
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv, Linear
# from torch.nn import CrossEntropyLoss, MarginRankingLoss
# from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer

# class GCN(torch.nn.Module):

#     def __init__(self, dim_in, dim_h, dim_out):
#         super().__init__()
#         self.gcn1 = GCNConv(dim_in, dim_h)
#         self.gcn2 = GCNConv(dim_h, dim_out)
    
#     def forward(self, x, edge_index):
#         x = self.gcn1(x, edge_index)
#         x = torch.relu(x)
#         x = F.dropout(x, p=0.5)
#         x = self.gcn2(x, edge_index)
#         return x, F.log_softmax(x, dim=1)

# class MAGDi(torch.nn.Module):

#     def __init__(self, base_model, gcn_in_channels, gcn_hidden_channels,
#                  gcn_out_channels, alpha, beta, gamma):
#         super(MAGDi, self).__init__()
#         # self.decoder = AutoModelForCausalLM.from_pretrained( model_name, cache_dir='./nas-ssd2/cychen/models')
#         self.decoder = base_model
#         self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
#         self.mlp1 = Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
#         self.mlp2 = Linear(self.decoder.config.hidden_size, 1)
#         self.mlp3 = Linear(self.decoder.config.vocab_size, 1)
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
        
#     def forward(self, pos_input_ids, pos_attention_mask, pos_labels, neg_input_ids, neg_attention_mask, neg_labels, graph):

#         device = pos_input_ids.device      # or next(self.gcn.parameters()).device, etc.

#         graph_loader = DataLoader(graph, batch_size=len(graph), shuffle=False, pin_memory=False, num_workers=0)
#         graph_batch = next(iter(graph_loader))
#         graph_batch = graph_batch.to(device)

#         pos_output = self.decoder(input_ids=pos_input_ids,
#                              attention_mask=pos_attention_mask,
#                              labels=pos_labels,
#                              output_hidden_states=True)

#         neg_output = self.decoder(input_ids=neg_input_ids,
#                              attention_mask=neg_attention_mask,
#                              labels=None,
#                              output_hidden_states=True)

#         row_sums = neg_attention_mask.sum(dim=1)
#         neg_mask = row_sums > 5 # ignore negative padding 
        
#         nll_loss = pos_output["loss"]
              
#         pos_last_hidden_state = pos_output.hidden_states[-1]
#         pos_weights_for_non_padding = pos_attention_mask * torch.arange(start=1, end=pos_last_hidden_state.shape[1] + 1).to(pos_attention_mask.device).unsqueeze(0)
#         pos_weights_for_non_padding = pos_weights_for_non_padding.to(pos_last_hidden_state.device)
#         pos_sum_embeddings = torch.sum(pos_last_hidden_state * pos_weights_for_non_padding.unsqueeze(-1), dim=1)
#         pos_num_of_none_padding_tokens = torch.sum(pos_weights_for_non_padding, dim=-1).unsqueeze(-1)
#         pos_seq_emb = pos_sum_embeddings / pos_num_of_none_padding_tokens

#         neg_last_hidden_state = neg_output.hidden_states[-1]
#         neg_weights_for_non_padding = neg_attention_mask * torch.arange(start=1, end=neg_last_hidden_state.shape[1] + 1).to(pos_attention_mask.device).unsqueeze(0)
#         neg_weights_for_non_padding = neg_weights_for_non_padding.to(neg_last_hidden_state.device)
#         neg_sum_embeddings = torch.sum(neg_last_hidden_state * neg_weights_for_non_padding.unsqueeze(-1), dim=1)
#         neg_num_of_none_padding_tokens = torch.sum(neg_weights_for_non_padding, dim=-1).unsqueeze(-1)
#         neg_seq_emb = neg_sum_embeddings / neg_num_of_none_padding_tokens
        
#         if neg_mask.any():
#             neg_mask = neg_mask.to(pos_seq_emb.device)
#             pos_seq_emb = pos_seq_emb[neg_mask]
#             neg_seq_emb = neg_seq_emb[neg_mask]
            
#         pos_h = torch.relu(self.mlp1(pos_seq_emb))
#         pos_score = self.mlp2(pos_h)
#         pos_score = torch.tanh(pos_score)
        
#         neg_h = torch.relu(self.mlp1(neg_seq_emb))
#         neg_score = self.mlp2(neg_h)
#         neg_score = torch.tanh(neg_score)
        
#         mr_cri = torch.nn.MarginRankingLoss(1.0, reduction='mean').to(pos_score.device)
#         mr_loss = mr_cri(pos_score, neg_score, torch.ones_like(pos_score).to(pos_score.device))
        
#         ce_cri = torch.nn.CrossEntropyLoss()
#         gcn_output, logits = self.gcn(graph_batch.x, graph_batch.edge_index)
#         graph_batch.y = graph_batch.y.to(logits.device)
#         node_loss = ce_cri(logits, graph_batch.y)
        
#         return self.alpha * nll_loss, self.beta * node_loss, self.gamma * mr_loss

# class MAGDiTrainer(Trainer):

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
      
#         output = model(pos_input_ids=inputs["pos_input_ids"],
#                        pos_attention_mask=inputs["pos_attention_mask"],
#                        pos_labels=inputs["pos_labels"],
#                        neg_input_ids=inputs["neg_input_ids"],
#                        neg_attention_mask=inputs["neg_attention_mask"],
#                        neg_labels=inputs["neg_labels"],
#                        graph=inputs["graph"])

#         nll_loss, node_loss, mr_loss = output
#         loss = nll_loss + node_loss + mr_loss

#         return loss
