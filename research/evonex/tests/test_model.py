"""
EvoMoE Model Tests
Validates architecture correctness without requiring trained weights.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_evonex import EvoMoE_Model, MultiScaleCNN_Expert, Transformer_Expert, PhysicsMLP_Expert


class TestEvoMoEArchitecture:
    """Tests for the complete EvoMoE model."""

    def test_forward_pass(self):
        model = EvoMoE_Model()
        lc = torch.randn(4, 2000)
        tic = torch.randn(4, 12)
        logits, weights = model(lc, tic)
        assert logits.shape == (4, 2)
        assert weights.shape == (4, 3)

    def test_single_sample(self):
        model = EvoMoE_Model()
        lc = torch.randn(1, 2000)
        tic = torch.randn(1, 12)
        logits, weights = model(lc, tic)
        assert logits.shape == (1, 2)

    def test_gating_weights_sum_to_one(self):
        model = EvoMoE_Model()
        lc = torch.randn(8, 2000)
        tic = torch.randn(8, 12)
        _, weights = model(lc, tic)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_gating_weights_non_negative(self):
        model = EvoMoE_Model()
        lc = torch.randn(4, 2000)
        tic = torch.randn(4, 12)
        _, weights = model(lc, tic)
        assert (weights >= 0).all()

    def test_parameter_count(self):
        model = EvoMoE_Model()
        total = sum(p.numel() for p in model.parameters())
        assert total > 1_000_000, f"Expected >1M params, got {total}"

    def test_eval_mode_deterministic(self):
        model = EvoMoE_Model()
        model.eval()
        torch.manual_seed(42)
        lc = torch.randn(2, 2000)
        tic = torch.randn(2, 12)
        with torch.no_grad():
            logits1, w1 = model(lc, tic)
            logits2, w2 = model(lc, tic)
        assert torch.equal(logits1, logits2)
        assert torch.equal(w1, w2)

    def test_custom_config(self):
        model = EvoMoE_Model(lc_seq_length=1000, num_tic_features=8, embed_dim=64, num_classes=3)
        lc = torch.randn(2, 1000)
        tic = torch.randn(2, 8)
        logits, weights = model(lc, tic)
        assert logits.shape == (2, 3)
        assert weights.shape == (2, 3)


class TestCNNExpert:
    def test_output_shape(self):
        expert = MultiScaleCNN_Expert(seq_len=2000, embed_dim=128)
        x = torch.randn(4, 1, 2000)
        emb, conf = expert(x)
        assert emb.shape == (4, 128)
        assert conf.shape == (4, 1)


class TestTransformerExpert:
    def test_output_shape(self):
        expert = Transformer_Expert(seq_len=2000, embed_dim=128)
        x = torch.randn(4, 1, 2000)
        emb, conf = expert(x)
        assert emb.shape == (4, 128)
        assert conf.shape == (4, 1)


class TestPhysicsExpert:
    def test_output_shape(self):
        expert = PhysicsMLP_Expert(num_tic_features=12, embed_dim=128)
        x = torch.randn(4, 12)
        emb, conf = expert(x)
        assert emb.shape == (4, 128)
        assert conf.shape == (4, 1)
