import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
)

class GCN(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1, self.gcn2 = GCNConv(dim_in, dim_h), GCNConv(dim_h, dim_out)
    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, 0.5, self.training)
        x = self.gcn2(x, edge_index)
        return x, F.log_softmax(x, dim=1)

class MAGDi(nn.Module):
    def __init__(
        self,
        model_name: str,
        gcn_in_channels: int,
        gcn_hidden_channels: int,
        gcn_out_channels: int,
        *,
        alpha: float = 1.0,
        beta:  float = 1.0,
        gamma: float = 0.1,
        eightbit: bool = False,
        device_map = None,
        use_gc: bool = False,
    ):
        super().__init__()

        if eightbit:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.decoder = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map=device_map or "auto",
            )
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device_map
            ).cuda()

        if use_gc:
            self.decoder.gradient_checkpointing_enable()

        self.gcn  = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
        self.mlp1 = Linear(self.decoder.config.hidden_size,
                           self.decoder.config.hidden_size)
        self.mlp2 = Linear(self.decoder.config.hidden_size, 1)

        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self._mr, self._ce = nn.MarginRankingLoss(1.0, "mean"), nn.CrossEntropyLoss()
        self.pad_token_id = self.decoder.config.pad_token_id or 0

    # ────────────────────────────────────────────────────────────────
    def forward(
        self,
        pos_input_ids, pos_attention_mask, pos_labels,
        neg_input_ids, neg_attention_mask, neg_labels,
        graph,
        return_breakdown=False,
    ):
        dev = pos_input_ids.device

        def _pad(t, L, pad_id):
            return t if t.size(1)==L else torch.cat([t, t.new_full((t.size(0), L-t.size(1)), pad_id)],1)

        # graph loss
        gb = graph.to(dev)
        _, g_logits = self.gcn(gb.x, gb.edge_index)
        node_loss = self._ce(g_logits, gb.y.to(dev))

        # pad to same length
        L = max(pos_input_ids.size(1), neg_input_ids.size(1))
        pos_input_ids      = _pad(pos_input_ids,      L, self.pad_token_id)
        neg_input_ids      = _pad(neg_input_ids,      L, self.pad_token_id)
        pos_attention_mask = _pad(pos_attention_mask, L, 0)
        neg_attention_mask = _pad(neg_attention_mask, L, 0)
        pos_labels         = _pad(pos_labels,         L, -100)
        neg_labels_padded  = neg_labels.new_full((neg_labels.size(0), L), -100)

        out = self.decoder(
            input_ids=torch.cat([pos_input_ids,  neg_input_ids], 0),
            attention_mask=torch.cat([pos_attention_mask, neg_attention_mask], 0),
            labels=torch.cat([pos_labels, neg_labels_padded], 0),
            output_hidden_states=True,
        )
        nll = out.loss
        hid_pos, hid_neg = out.hidden_states[-1].chunk(2,0)

        def mean_pool(h, m):
            w = m.to(h.device) * torch.arange(1,h.size(1)+1,device=h.device).unsqueeze(0)
            return (h*w.unsqueeze(-1)).sum(1)/w.sum(1,keepdim=True)

        pos_emb = mean_pool(hid_pos, pos_attention_mask)
        neg_emb = mean_pool(hid_neg, neg_attention_mask)
        valid   = (neg_attention_mask.sum(1)>5).to(dev)
        mr = self._mr(
            torch.tanh(self.mlp2(F.relu(self.mlp1(pos_emb[valid])))),
            torch.tanh(self.mlp2(F.relu(self.mlp1(neg_emb[valid])))),
            torch.ones(valid.sum(),1,device=dev)
        ) if valid.any() else nll.new_tensor(0.0)

        total = self.alpha*nll + self.beta*node_loss + self.gamma*mr
        if return_breakdown:
            return total, {"nll":nll.detach(),"node":node_loss.detach(),"mr":mr.detach()}
        return total

class MAGDiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            loss, parts = model(return_breakdown=True, **inputs)
            return loss, parts
        return model(**inputs)



##################
##################
# # model_fused.py – Fuse v1.4 (final)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, Linear
# from transformers import AutoModelForCausalLM, Trainer

# # import torch
# # torch.cuda.set_device(0)           # ← pick GPU id 0 for the whole process


# # ─────────────────────────  Graph branch  ────────────────────────────
# class GCN(nn.Module):
#     def __init__(self, dim_in: int, dim_h: int, dim_out: int):
#         super().__init__()
#         self.gcn1, self.gcn2 = GCNConv(dim_in, dim_h), GCNConv(dim_h, dim_out)

#     def forward(self, x, edge_index):
#         x = self.gcn1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, 0.5, self.training)
#         x = self.gcn2(x, edge_index)
#         return x, F.log_softmax(x, dim=1)


# # ─────────────────────────────  MAGDi  ───────────────────────────────
# class MAGDi(nn.Module):
#     def __init__(
#         self,
#         model_name: str,
#         gcn_in_channels: int,
#         gcn_hidden_channels: int,
#         gcn_out_channels: int,
#         alpha: float = 1.0,
#         beta:  float = 1.0,
#         gamma: float = 0.1,
#         *,
#         load_in_8bit: bool = False,
#         use_gc: bool = False,          # gradient-checkpointing flag
#     ):
#         super().__init__()

#         self.decoder = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             load_in_8bit=load_in_8bit,
#             device_map="auto" if load_in_8bit else None,
#         )
#         if use_gc:
#             self.decoder.gradient_checkpointing_enable()

#         self.gcn  = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
#         self.mlp1 = Linear(self.decoder.config.hidden_size,
#                            self.decoder.config.hidden_size)
#         self.mlp2 = Linear(self.decoder.config.hidden_size, 1)

#         self.alpha, self.beta, self.gamma = alpha, beta, gamma
#         self._mr, self._ce = nn.MarginRankingLoss(1.0, "mean"), nn.CrossEntropyLoss()

#         # fall-back pad ID if backbone doesn’t define one
#         self.pad_token_id = (self.decoder.config.pad_token_id
#                              if self.decoder.config.pad_token_id is not None
#                              else 0)

#     # ────────────────────────────────────────────────────────────────
#     def forward(
#         self,
#         pos_input_ids, pos_attention_mask, pos_labels,
#         neg_input_ids, neg_attention_mask, neg_labels,
#         graph,
#         return_breakdown: bool = False,
#     ):
#         dev = pos_input_ids.device

#         # 0) helper: right-pad [B, L] tensor to target_len
#         def _pad(t, target_len, pad_id):
#             if t.size(1) == target_len:
#                 return t
#             pad = t.new_full((t.size(0), target_len - t.size(1)), pad_id)
#             return torch.cat([t, pad], 1)

#         # 1) graph-classification loss
#         gb = graph.to(dev)
#         _, g_logits = self.gcn(gb.x, gb.edge_index)
#         node_loss   = self._ce(g_logits, gb.y.to(dev))

#         # 2) ensure pos & neg have same seq-len
#         max_len = max(pos_input_ids.size(1), neg_input_ids.size(1))

#         pos_input_ids      = _pad(pos_input_ids,      max_len, self.pad_token_id)
#         neg_input_ids      = _pad(neg_input_ids,      max_len, self.pad_token_id)
#         pos_attention_mask = _pad(pos_attention_mask, max_len, 0)
#         neg_attention_mask = _pad(neg_attention_mask, max_len, 0)
#         pos_labels         = _pad(pos_labels,         max_len, -100)

#         # pad the *neg_labels* as well (all -100 so they are ignored)
#         neg_labels_padded  = neg_labels.new_full((neg_labels.size(0), max_len), -100)

#         # ── single decoder pass ──────────────────────────────────────────────
#         ids    = torch.cat([pos_input_ids,  neg_input_ids], 0)
#         masks  = torch.cat([pos_attention_mask, neg_attention_mask], 0)
#         labels = torch.cat([pos_labels, neg_labels_padded], 0)

#         dec_out  = self.decoder(input_ids=ids,
#                                 attention_mask=masks,
#                                 labels=labels,
#                                 output_hidden_states=True)
#         nll_loss = dec_out.loss
#         hid_pos, hid_neg = dec_out.hidden_states[-1].chunk(2, 0)

#         # 4) margin-ranking loss
#         pos_emb = self._mean_pool(hid_pos, pos_attention_mask)
#         neg_emb = self._mean_pool(hid_neg, neg_attention_mask)
#         valid   = (neg_attention_mask.sum(1) > 5).to(dev)
#         mr_loss = (
#             self._mr(
#                 torch.tanh(self.mlp2(F.relu(self.mlp1(pos_emb[valid])))),
#                 torch.tanh(self.mlp2(F.relu(self.mlp1(neg_emb[valid])))),
#                 torch.ones(valid.sum(), 1, device=dev),
#             )
#             if valid.any()
#             else nll_loss.new_tensor(0.0)
#         )

#         total = self.alpha * nll_loss + self.beta * node_loss + self.gamma * mr_loss
#         if return_breakdown:
#             return total, {"nll": nll_loss.detach(),
#                            "node": node_loss.detach(),
#                            "mr":  mr_loss.detach()}
#         return total

#     @staticmethod
#     def _mean_pool(hidden, mask):
#         mask = mask.to(hidden.device)
#         w    = mask * torch.arange(1, hidden.size(1) + 1,
#                                    device=hidden.device).unsqueeze(0)
#         return (hidden * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True)


# # ────────────────────────────  Trainer  ──────────────────────────────
# class MAGDiTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         if return_outputs:
#             loss, parts = model(return_breakdown=True, **inputs)
#             return loss, parts
#         return model(**inputs)

##################
######
# # model_fused.py – Fuse v1.3
# import torch, torch.nn as nn, torch.nn.functional as F
# from torch_geometric.nn import GCNConv, Linear
# from transformers import AutoModelForCausalLM, Trainer


# # ─────────────────────────  Graph branch  ────────────────────────────
# class GCN(nn.Module):
#     def __init__(self, dim_in: int, dim_h: int, dim_out: int):
#         super().__init__()
#         self.gcn1, self.gcn2 = GCNConv(dim_in, dim_h), GCNConv(dim_h, dim_out)

#     def forward(self, x, edge_index):
#         x = self.gcn1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, 0.5, self.training)
#         x = self.gcn2(x, edge_index)
#         return x, F.log_softmax(x, dim=1)


# # ─────────────────────────────  MAGDi  ───────────────────────────────
# class MAGDi(nn.Module):
#     def __init__(
#         self,
#         model_name: str,
#         gcn_in_channels: int,
#         gcn_hidden_channels: int,
#         gcn_out_channels: int,
#         alpha: float = 1.0,
#         beta:  float = 1.0,
#         gamma: float = 0.1,
#         *,
#         load_in_8bit: bool = False,
#         use_gc: bool = False,          # gradient-checkpointing flag
#     ):
#         super().__init__()

#         self.decoder = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             load_in_8bit=load_in_8bit,
#             device_map="auto" if load_in_8bit else None,
#         )
#         if use_gc:
#             self.decoder.gradient_checkpointing_enable()

#         self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
#         self.mlp1 = Linear(self.decoder.config.hidden_size,
#                            self.decoder.config.hidden_size)
#         self.mlp2 = Linear(self.decoder.config.hidden_size, 1)

#         self.alpha, self.beta, self.gamma = alpha, beta, gamma
#         self._mr, self._ce = nn.MarginRankingLoss(1.0, "mean"), nn.CrossEntropyLoss()

#     # ────────────────────────────────────────────────────────────────
#     def forward(
#         self,
#         pos_input_ids, pos_attention_mask, pos_labels,
#         neg_input_ids, neg_attention_mask, neg_labels,
#         graph,
#         return_breakdown: bool = False,
#     ):
#         dev = pos_input_ids.device

#         # 1) Graph loss
#         gb = graph.to(dev)
#         _, g_logits = self.gcn(gb.x, gb.edge_index)
#         node_loss   = self._ce(g_logits, gb.y.to(dev))

#         # 2) Single decoder pass for pos+neg
#         ids   = torch.cat([pos_input_ids,  neg_input_ids],  0)
#         masks = torch.cat([pos_attention_mask, neg_attention_mask], 0)
#         labels= torch.cat([pos_labels,
#                            pos_labels.new_full(neg_labels.size(), -100)], 0)  # ignore loss on neg part

#         dec_out  = self.decoder(input_ids=ids,
#                                 attention_mask=masks,
#                                 labels=labels,
#                                 output_hidden_states=True)
#         nll_loss = dec_out.loss
#         hid_pos, hid_neg = dec_out.hidden_states[-1].chunk(2, 0)

#         # 3) Margin-ranking
#         pos_emb = self._mean_pool(hid_pos, pos_attention_mask)
#         neg_emb = self._mean_pool(hid_neg, neg_attention_mask)
#         valid   = (neg_attention_mask.sum(1) > 5).to(dev)
#         mr_loss = (
#             self._mr(
#                 torch.tanh(self.mlp2(F.relu(self.mlp1(pos_emb[valid])))),
#                 torch.tanh(self.mlp2(F.relu(self.mlp1(neg_emb[valid])))),
#                 torch.ones(valid.sum(), 1, device=dev),
#             )
#             if valid.any()
#             else nll_loss.new_tensor(0.0)
#         )

#         total = self.alpha * nll_loss + self.beta * node_loss + self.gamma * mr_loss
#         if return_breakdown:
#             return total, {"nll": nll_loss.detach(),
#                            "node": node_loss.detach(),
#                            "mr":  mr_loss.detach()}
#         return total

#     @staticmethod
#     def _mean_pool(hidden, mask):
#         mask = mask.to(hidden.device)
#         w    = mask * torch.arange(1, hidden.size(1)+1, device=hidden.device).unsqueeze(0)
#         return (hidden * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True)


# # ────────────────────────────  Trainer  ──────────────────────────────
# class MAGDiTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         if return_outputs:
#             loss, parts = model(return_breakdown=True, **inputs)
#             return loss, parts
#         return model(**inputs)

#########


# # model_fused.py  –  Fuse v1.2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, Linear
# from transformers import AutoModelForCausalLM, Trainer


# # ─────────────────────────  Graph sub-module  ──────────────────────────
# class GCN(nn.Module):
#     def __init__(self, dim_in: int, dim_h: int, dim_out: int):
#         super().__init__()
#         self.gcn1 = GCNConv(dim_in, dim_h)
#         self.gcn2 = GCNConv(dim_h, dim_out)

#     def forward(self, x, edge_index):
#         x = self.gcn1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.gcn2(x, edge_index)
#         return x, F.log_softmax(x, dim=1)


# # ─────────────────────────────  MAGDi  ─────────────────────────────────
# class MAGDi(nn.Module):
#     def __init__(
#         self,
#         model_name: str,
#         gcn_in_channels: int,
#         gcn_hidden_channels: int,
#         gcn_out_channels: int,
#         alpha: float = 1.0,
#         beta: float = 1.0,
#         gamma: float = 0.1,
#     ):
#         super().__init__()
#         self.decoder = AutoModelForCausalLM.from_pretrained(model_name)
#         self.gcn     = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)

#         self.mlp1 = Linear(self.decoder.config.hidden_size,
#                            self.decoder.config.hidden_size)
#         self.mlp2 = Linear(self.decoder.config.hidden_size, 1)

#         self.alpha, self.beta, self.gamma = alpha, beta, gamma
#         self._mr = nn.MarginRankingLoss(1.0, reduction="mean")
#         self._ce = nn.CrossEntropyLoss()

#     # ------------------------------------------------------------------
#     def forward(
#         self,
#         pos_input_ids, pos_attention_mask, pos_labels,
#         neg_input_ids, neg_attention_mask, neg_labels,
#         graph,
#         return_breakdown: bool = False,
#     ):
#         device = pos_input_ids.device

#         # 1) Graph branch
#         g_batch = graph.to(device)
#         _, g_logits = self.gcn(g_batch.x, g_batch.edge_index)
#         node_loss = self._ce(g_logits, g_batch.y.to(device))

#         # 2) Positive sequence
#         pos_out  = self.decoder(input_ids=pos_input_ids,
#                                 attention_mask=pos_attention_mask,
#                                 labels=pos_labels,
#                                 output_hidden_states=True)
#         nll_loss = pos_out.loss

#         # 3) Negative sequence
#         neg_out = self.decoder(input_ids=neg_input_ids,
#                                attention_mask=neg_attention_mask,
#                                output_hidden_states=True)

#         # 4) Mean-pool embeddings
#         pos_all = self._mean_pool(pos_out.hidden_states[-1], pos_attention_mask)
#         neg_all = self._mean_pool(neg_out.hidden_states[-1], neg_attention_mask)

#         valid = neg_attention_mask.sum(1) > 5
#         if valid.any():
#             pos_emb = pos_all[valid.to(device)]
#             neg_emb = neg_all[valid.to(device)]

#             pos_score = torch.tanh(self.mlp2(F.relu(self.mlp1(pos_emb))))
#             neg_score = torch.tanh(self.mlp2(F.relu(self.mlp1(neg_emb))))
#             mr_loss   = self._mr(pos_score, neg_score,
#                                  torch.ones_like(pos_score))
#         else:
#             mr_loss = nll_loss.new_tensor(0.0)

#         total = self.alpha * nll_loss + self.beta * node_loss + self.gamma * mr_loss
#         if return_breakdown:
#             return total, {"nll": nll_loss.detach(),
#                            "node": node_loss.detach(),
#                            "mr": mr_loss.detach()}
#         return total

#     @staticmethod
#     def _mean_pool(hidden, mask):
#         mask = mask.to(hidden.device)
#         w    = mask * torch.arange(1, hidden.size(1) + 1,
#                                    device=hidden.device).unsqueeze(0)
#         return (hidden * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True)


# # ─────────────────────────────  Trainer  ───────────────────────────────
# class MAGDiTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         if return_outputs:
#             loss, parts = model(return_breakdown=True, **inputs)
#             return loss, parts
#         return model(**inputs)
