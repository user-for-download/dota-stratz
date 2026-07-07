"""PyTorch Transformer Architecture for Dota 2 Drafts (DraftBERT).

Multi-Modal architecture combining:
- Transformer encoder for draft sequence (hero picks/bans)
- MLP for continuous tabular features (team/player aggregates)
- Fusion head for final win probability prediction
"""

import torch
import torch.nn as nn


class MultiModalDraftBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 165,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_continuous_features: int = 59,
        max_seq_len: int = 50,
        dropout: float = 0.3,
        transformer_dropout: float = 0.1,
        fusion_hidden: int = 64,
    ):
        super().__init__()

        # --- 1. Sequence Branch (Transformer) ---
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

        # --- 2. Tabular Branch (Continuous Features) ---
        self.tabular_mlp = nn.Sequential(
            nn.LayerNorm(num_continuous_features),
            nn.Linear(num_continuous_features, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- 3. Fusion Head ---
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model + fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, heroes, actions, continuous_features):
        """
        heroes: (Batch, SeqLen) — hero IDs, 0 = padding
        actions: (Batch, SeqLen) — action tokens (0=pad, 1=RadBan, 2=DireBan, 3=RadPick, 4=DirePick)
        continuous_features: (Batch, num_continuous) — tabular aggregate features
        Returns: (Batch,) — raw logits for Radiant win probability
        """
        B, S = heroes.size()
        positions = torch.arange(S, device=heroes.device).unsqueeze(0).expand(B, S)

        # Sequence processing with embedding dropout
        x = self.hero_emb(heroes) + self.action_emb(actions) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        pad_mask = (heroes == 0)
        out = self.transformer(x, src_key_padding_mask=pad_mask)

        # Mean pooling (ignore padding)
        mask_expanded = pad_mask.unsqueeze(-1).expand_as(out)
        out = out.masked_fill(mask_expanded, 0.0)
        sum_embeddings = out.sum(dim=1)
        valid_lengths = (~pad_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        seq_repr = sum_embeddings / valid_lengths

        # Tabular processing
        tab_repr = self.tabular_mlp(continuous_features)

        # Fusion and prediction
        fused = torch.cat([seq_repr, tab_repr], dim=1)
        logits = self.fusion_head(fused).view(-1)
        return logits
