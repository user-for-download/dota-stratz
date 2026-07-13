"""Tests for PyTorch model architectures (DraftBERT + LiveDraftBERT).

Verifies forward pass shapes, NaN safety, and transformer masking behavior.
"""

import torch
import pytest

from trainer.model_pt import MultiModalDraftBERT
from trainer.model_live import LiveDraftBERT


class TestMultiModalDraftBERT:
    def test_basic_forward_shape(self):
        model = MultiModalDraftBERT(
            vocab_size=165, d_model=16, nhead=2, num_layers=1,
            num_continuous_features=61, max_seq_len=50
        )
        heroes = torch.randint(1, 160, (4, 5))
        actions = torch.randint(1, 5, (4, 5))
        tabular = torch.randn((4, 61))
        patches = torch.randint(55, 61, (4,))
        logits = model(heroes, actions, tabular, patches)
        assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
        assert not torch.isnan(logits).any(), "Model produced NaN"

    def test_single_sample(self):
        model = MultiModalDraftBERT(num_continuous_features=61)
        h = torch.tensor([[14, 53, 0, 0, 0]])
        a = torch.tensor([[3, 4, 0, 0, 0]])
        f = torch.zeros((1, 61))
        p = torch.tensor([60])
        out = model(h, a, f, p)
        assert out.shape == (1,)

    def test_full_draft(self):
        model = MultiModalDraftBERT(num_continuous_features=61, max_seq_len=50)
        h = torch.randint(1, 160, (2, 50))
        a = torch.randint(1, 5, (2, 50))
        f = torch.randn((2, 61))
        p = torch.tensor([60, 59])
        out = model(h, a, f, p)
        assert out.shape == (2,)

    def test_all_padding(self):
        """Model should handle all-zero (all padding) input without crash."""
        model = MultiModalDraftBERT(num_continuous_features=61)
        h = torch.zeros((1, 50), dtype=torch.long)
        a = torch.zeros((1, 50), dtype=torch.long)
        f = torch.zeros((1, 61))
        p = torch.tensor([0])
        out = model(h, a, f, p)
        assert out.shape == (1,)
        assert not torch.isnan(out).any()

    def test_eval_mode(self):
        model = MultiModalDraftBERT(num_continuous_features=61)
        model.eval()
        h = torch.randint(1, 160, (1, 10))
        a = torch.randint(1, 5, (1, 10))
        f = torch.randn((1, 61))
        p = torch.tensor([60])
        with torch.no_grad():
            out1 = model(h, a, f, p)
            out2 = model(h, a, f, p)
        assert torch.allclose(out1, out2), "eval mode should be deterministic"


class TestLiveDraftBERT:
    def test_basic_forward_shape(self):
        model = LiveDraftBERT(
            num_static_features=61, num_dynamic_features=30
        )
        heroes = torch.randint(1, 160, (2, 24))
        actions = torch.randint(1, 5, (2, 24))
        static = torch.randn((2, 61))
        dynamic = torch.randn((2, 30))
        patches = torch.tensor([60, 59])
        out = model(heroes, actions, static, dynamic, patches)
        assert out.shape == (2,), f"Expected (2,), got {out.shape}"
        assert not torch.isnan(out).any()

    def test_single_sample(self):
        model = LiveDraftBERT(num_static_features=61, num_dynamic_features=30)
        h = torch.tensor([[14, 53, 0, 0, 0]])
        a = torch.tensor([[3, 4, 0, 0, 0]])
        s = torch.zeros((1, 61))
        d = torch.randn((1, 30))
        p = torch.tensor([60])
        out = model(h, a, s, d, p)
        assert out.shape == (1,)

    def test_dynamic_features_matter(self):
        """Changing only dynamic features should change the output."""
        model = LiveDraftBERT(num_static_features=61, num_dynamic_features=30)
        model.eval()
        h = torch.randint(1, 160, (1, 10))
        a = torch.randint(1, 5, (1, 10))
        s = torch.randn((1, 61))
        d1 = torch.zeros((1, 30))
        d2 = torch.ones((1, 30))
        p = torch.tensor([60])
        with torch.no_grad():
            out1 = model(h, a, s, d1, p)
            out2 = model(h, a, s, d2, p)
        assert not torch.allclose(out1, out2), "Dynamic features should affect output"
