"""LiveDraftBERT: Extended Multi-Modal architecture for live match prediction.

Adds a Dynamic MLP branch to the existing DraftBERT architecture:
- Branch 1: Transformer (draft sequence) → 128-dim [CACHED per match]
- Branch 2: Static MLP (61 pre-game aggregates) → 64-dim [CACHED per match]
- Branch 3: Dynamic MLP (35 live game state features) → 32-dim [RE-EVALUATED per tick]
- Fusion Head: Linear(128+64+32, 64) → ReLU → Dropout → Linear(64, 1)
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
        num_dynamic_features: int = 35,
        max_seq_len: int = 25,
        dropout: float = 0.3,
        transformer_dropout: float = 0.1,
        static_hidden: int = 64,
        dynamic_hidden: int = 32,
        fusion_hidden: int = 64,
        max_patch_id: int = 200,
    ):
        super().__init__()

        # Branch 1: Transformer
        self.hero_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.action_emb = nn.Embedding(5, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=transformer_dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for stable gradients
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Branch 2: Static MLP (2-layer residual)
        tabular_dropout = min(0.5, dropout * 1.5)
        self.static_norm = nn.LayerNorm(num_static_features)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden),
            nn.GELU(),
            nn.Dropout(tabular_dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.GELU(),
            nn.Dropout(tabular_dropout),
        )

        # Branch 3: Dynamic MLP (2-layer residual)
        self.dynamic_norm = nn.LayerNorm(num_dynamic_features)
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(num_dynamic_features, dynamic_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dynamic_hidden, dynamic_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Branch 4: Patch Embedding
        self.patch_emb = nn.Embedding(max_patch_id, fusion_hidden, padding_idx=0)

        # Fusion Head (no gate — straight concatenation)
        fusion_dim = d_model + static_hidden + dynamic_hidden + fusion_hidden
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, heroes, actions, static_features, dynamic_features, patches):
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

        static_repr = self.static_mlp(self.static_norm(static_features))
        dynamic_repr = self.dynamic_mlp(self.dynamic_norm(dynamic_features))
        patch_repr = self.patch_emb(patches)

        fused = torch.cat([seq_repr, static_repr, dynamic_repr, patch_repr], dim=1)
        logits = self.fusion_head(fused).squeeze(-1)
        return logits

    def forward_dynamic(self, seq_repr, static_repr, dynamic_features, patches):
        dynamic_repr = self.dynamic_mlp(self.dynamic_norm(dynamic_features))
        patch_repr = self.patch_emb(patches)
        fused = torch.cat([seq_repr, static_repr, dynamic_repr, patch_repr], dim=1)
        logits = self.fusion_head(fused).squeeze(-1)
        return logits

    def encode_draft(self, heroes, actions, static_features, patches):
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

        static_repr = self.static_mlp(self.static_norm(static_features))
        patch_repr = self.patch_emb(patches)
        return seq_repr, static_repr, patch_repr
