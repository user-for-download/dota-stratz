"""LiveDraftBERT: Extended Multi-Modal architecture for live match prediction.

Mirrors the trainer's model_live.py exactly. Loaded via state_dict
(instead of TorchScript) to avoid TransformerEncoderLayer trace failures.
"""

import torch
import torch.nn as nn


class LiveDraftBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 165,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_static_features: int = 59,
        num_dynamic_features: int = 24,
        max_seq_len: int = 50,
        dropout: float = 0.3,
        transformer_dropout: float = 0.1,
        static_hidden: int = 64,
        dynamic_hidden: int = 32,
        fusion_hidden: int = 64,
    ):
        super().__init__()

        self.hero_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.action_emb = nn.Embedding(5, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.static_mlp = nn.Sequential(
            nn.LayerNorm(num_static_features),
            nn.Linear(num_static_features, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dynamic_mlp = nn.Sequential(
            nn.LayerNorm(num_dynamic_features),
            nn.Linear(num_dynamic_features, dynamic_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(d_model + static_hidden + dynamic_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, heroes, actions, static_features, dynamic_features):
        B, S = heroes.size()
        positions = torch.arange(S, device=heroes.device).unsqueeze(0).expand(B, S)

        x = self.hero_emb(heroes) + self.action_emb(actions) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        pad_mask = (heroes == 0)
        out = self.transformer(x, src_key_padding_mask=pad_mask)

        mask_expanded = pad_mask.unsqueeze(-1).expand_as(out)
        out = out.masked_fill(mask_expanded, 0.0)
        sum_embeddings = out.sum(dim=1)
        valid_lengths = (~pad_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        seq_repr = sum_embeddings / valid_lengths

        static_repr = self.static_mlp(static_features)
        dynamic_repr = self.dynamic_mlp(dynamic_features)

        fused = torch.cat([seq_repr, static_repr, dynamic_repr], dim=1)
        logits = self.fusion_head(fused).squeeze(-1)
        return logits

    def forward_dynamic(self, seq_repr, static_repr, dynamic_features):
        dynamic_repr = self.dynamic_mlp(dynamic_features)
        fused = torch.cat([seq_repr, static_repr, dynamic_repr], dim=1)
        logits = self.fusion_head(fused).squeeze(-1)
        return logits

    def encode_draft(self, heroes, actions, static_features):
        B, S = heroes.size()
        positions = torch.arange(S, device=heroes.device).unsqueeze(0).expand(B, S)

        x = self.hero_emb(heroes) + self.action_emb(actions) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        pad_mask = (heroes == 0)
        out = self.transformer(x, src_key_padding_mask=pad_mask)

        mask_expanded = pad_mask.unsqueeze(-1).expand_as(out)
        out = out.masked_fill(mask_expanded, 0.0)
        sum_embeddings = out.sum(dim=1)
        valid_lengths = (~pad_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        seq_repr = sum_embeddings / valid_lengths

        static_repr = self.static_mlp(static_features)
        return seq_repr, static_repr
