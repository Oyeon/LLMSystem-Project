# model_original.py  –  unfused baseline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Linear
from transformers import AutoModelForCausalLM, Trainer


# ─────────────────────────  Graph sub-module  ──────────────────────────
class GCN(nn.Module):
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


# ─────────────────────────────  MAGDi  ─────────────────────────────────
class MAGDi(nn.Module):
    def __init__(
        self,
        model_name,
        gcn_in_channels,
        gcn_hidden_channels,
        gcn_out_channels,
        alpha,
        beta,
        gamma,
    ):
        super().__init__()
        self.decoder = AutoModelForCausalLM.from_pretrained(model_name)
        self.gcn     = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)

        self.mlp1 = Linear(self.decoder.config.hidden_size,
                           self.decoder.config.hidden_size)
        self.mlp2 = Linear(self.decoder.config.hidden_size, 1)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    # ------------------------------------------------------------------
    def forward(
        self,
        pos_input_ids, pos_attention_mask, pos_labels,
        neg_input_ids, neg_attention_mask, neg_labels,
        graph
    ):
        device = pos_input_ids.device

        # 1) Graph loader (one extra DataLoader pass each step)
        graph_batch = next(iter(DataLoader(graph,
                                           batch_size=len(graph),
                                           shuffle=False,
                                           num_workers=0))).to(device)
        gcn_out, logits = self.gcn(graph_batch.x, graph_batch.edge_index)
        node_loss = nn.CrossEntropyLoss()(logits, graph_batch.y.to(device))

        # 2) Positive pass  (decoder forward + loss)
        pos_out  = self.decoder(input_ids=pos_input_ids,
                                attention_mask=pos_attention_mask,
                                labels=pos_labels,
                                output_hidden_states=True)
        nll_loss = pos_out.loss

        # 3) Negative pass  (decoder forward only)
        neg_out = self.decoder(input_ids=neg_input_ids,
                               attention_mask=neg_attention_mask,
                               labels=None,
                               output_hidden_states=True)

        # 4) Margin-Ranking loss
        pos_emb = self._mean_pool(pos_out.hidden_states[-1], pos_attention_mask)
        neg_emb = self._mean_pool(neg_out.hidden_states[-1], neg_attention_mask)

        row_sums = neg_attention_mask.sum(1)
        mask     = row_sums > 5
        if mask.any():
            pos_emb = pos_emb[mask.to(device)]
            neg_emb = neg_emb[mask.to(device)]

        pos_score = torch.tanh(self.mlp2(torch.relu(self.mlp1(pos_emb))))
        neg_score = torch.tanh(self.mlp2(torch.relu(self.mlp1(neg_emb))))

        mr_loss = nn.MarginRankingLoss(1.0, reduction="mean")(
            pos_score, neg_score, torch.ones_like(pos_score))

        return (
            self.alpha * nll_loss,
            self.beta  * node_loss,
            self.gamma * mr_loss,
        )

    @staticmethod
    def _mean_pool(hidden, mask):
        mask = mask.to(hidden.device)
        weights = mask * torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device).unsqueeze(0)
        return (hidden * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True)


# ─────────────────────────────  Trainer  ───────────────────────────────
class MAGDiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        nll, node, mr = model(
            pos_input_ids       = inputs["pos_input_ids"],
            pos_attention_mask  = inputs["pos_attention_mask"],
            pos_labels          = inputs["pos_labels"],
            neg_input_ids       = inputs["neg_input_ids"],
            neg_attention_mask  = inputs["neg_attention_mask"],
            neg_labels          = inputs["neg_labels"],
            graph               = inputs["graph"],
        )
        loss = nll + node + mr
        return (loss, (nll, node, mr)) if return_outputs else loss
