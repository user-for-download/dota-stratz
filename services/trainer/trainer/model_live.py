"""LiveDraftBERT: Extended Multi-Modal architecture for live match prediction.

Adds a Dynamic MLP branch to the existing DraftBERT architecture:
- Branch 1: Transformer (draft sequence) → 128-dim [CACHED per match]
- Branch 2: Static MLP (59 pre-game aggregates) → 64-dim [CACHED per match]
- Branch 3: Dynamic MLP (15 live game state features) → 32-dim [RE-EVALUATED per tick]
- Fusion Head: Linear(128+64+32, 64) → ReLU → Dropout → Linear(64, 1)

The Transformer and Static MLP embeddings are computed once at match start
and cached. Only the Dynamic MLP runs every 30s, reducing per-tick cost
from ~5ms to ~0.1ms.
"""

import torch
import torch.nn as nn


class LiveDraftBERT(nn.Module):
    """Extended DraftBERT with Dynamic MLP branch for live game state."""

    def __init__(
        self,
        vocab_size: int = 165,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_static_features: int = 59,
        num_dynamic_features: int = 15,
        max_seq_len: int = 50,
    ):
        super().__init__()

        # --- Branch 1: Transformer (draft sequence) ---
        self.hero_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.action_emb = nn.Embedding(5, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(0.3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Branch 2: Static MLP (pre-game aggregates) ---
        self.static_mlp = nn.Sequential(
            nn.LayerNorm(num_static_features),
            nn.Linear(num_static_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # --- Branch 3: Dynamic MLP (live game state) ---
        self.dynamic_mlp = nn.Sequential(
            nn.LayerNorm(num_dynamic_features),
            nn.Linear(num_dynamic_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # --- Fusion Head (128 + 64 + 32 = 224 → 64 → 1) ---
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model + 64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, heroes, actions, static_features, dynamic_features):
        """
        Full forward pass (used at minute 0, caches embeddings).

        heroes: (B, SeqLen) — draft hero IDs
        actions: (B, SeqLen) — action tokens (0=pad, 1=RadBan, 2=DireBan, 3=RadPick, 4=DirePick)
        static_features: (B, num_static) — pre-game aggregates
        dynamic_features: (B, num_dynamic) — live game state features
        Returns: (B,) — raw logits for Radiant win probability
        """
        B, S = heroes.size()
        positions = torch.arange(S, device=heroes.device).unsqueeze(0).expand(B, S)

        # Branch 1: Transformer
        x = self.hero_emb(heroes) + self.action_emb(actions) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        pad_mask = (heroes == 0)
        out = self.transformer(x, src_key_padding_mask=pad_mask)

        mask_expanded = pad_mask.unsqueeze(-1).expand_as(out)
        out = out.masked_fill(mask_expanded, 0.0)
        sum_embeddings = out.sum(dim=1)
        valid_lengths = (~pad_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        seq_repr = sum_embeddings / valid_lengths  # (B, 128)

        # Branch 2: Static MLP
        static_repr = self.static_mlp(static_features)  # (B, 64)

        # Branch 3: Dynamic MLP
        dynamic_repr = self.dynamic_mlp(dynamic_features)  # (B, 32)

        # Fusion
        fused = torch.cat([seq_repr, static_repr, dynamic_repr], dim=1)  # (B, 224)
        logits = self.fusion_head(fused).squeeze(-1)  # (B,)
        return logits

    def forward_dynamic(self, seq_repr, static_repr, dynamic_features):
        """Fast inference path using cached transformer + static embeddings.

        seq_repr: (B, 128) — cached transformer output
        static_repr: (B, 64) — cached static MLP output
        dynamic_features: (B, num_dynamic) — new live features (changes every tick)
        Returns: (B,) — raw logits
        """
        dynamic_repr = self.dynamic_mlp(dynamic_features)  # (B, 32)
        fused = torch.cat([seq_repr, static_repr, dynamic_repr], dim=1)  # (B, 224)
        logits = self.fusion_head(fused).squeeze(-1)  # (B,)
        return logits

    def encode_draft(self, heroes, actions, static_features):
        """Encode draft + static features for caching (called once per match).

        Returns:
            seq_repr: (B, 128) — transformer embedding
            static_repr: (B, 64) — static MLP embedding
        """
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
