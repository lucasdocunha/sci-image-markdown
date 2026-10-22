"""
Unit and integration tests for ICDAR Metric Loss (SCST and Token-Weighted).
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from src.training.icdar_loss import (
    ICDARTokenWeightedLoss,
    ICDARSCSTLoss,
    ICDARMetricLoss,
)


class DummyTokenizer:
    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<eos>": 1,
            "|": 2,
            "-": 3,
            "1": 4,
            "2": 5,
            ".": 6,
            "A": 7,
            "B": 8,
            "\n": 9,
        }
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.get(ch, 0) for ch in text if ch in self.vocab]

    def decode(self, token_ids, skip_special_tokens=True):
        rev = {v: k for k, v in self.vocab.items()}
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(rev.get(idx, "") for idx in token_ids)


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class DummyModel(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.lm_head = nn.Linear(16, vocab_size)

    def forward(self, input_ids, **kwargs):
        h = self.embedding(input_ids)
        logits = self.lm_head(h)
        loss = None
        if "labels" in kwargs and kwargs["labels"] is not None:
            labels = kwargs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        # Deterministically appends tokens
        new_tokens = torch.tensor([[2, 4, 2, 5, 2, 1]], device=input_ids.device)
        return torch.cat([input_ids, new_tokens], dim=1)


def test_token_weighted_loss_gradient_flow():
    tokenizer = DummyTokenizer()
    loss_module = ICDARTokenWeightedLoss(tokenizer, numeric_weight=3.0, structural_weight=2.0)

    logits = torch.randn(2, 6, len(tokenizer), requires_grad=True)
    labels = torch.tensor([
        [-100, -100, 2, 4, 6, 1],
        [-100, 7, 2, 8, 9, 1],
    ])

    loss = loss_module(logits, labels)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_scst_loss_forward_and_backward():
    processor = DummyProcessor()
    scst_module = ICDARSCSTLoss(processor, max_gen_tokens=10)
    model = DummyModel(len(processor.tokenizer))

    inputs = {
        "input_ids": torch.tensor([[7, 8, 2, 4, 2, 5, 2, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 2, 4, 2, 5, 2, 1]]),
    }

    loss, stats = scst_module(model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert "mean_r_sampled" in stats
    assert "mean_r_greedy" in stats
    assert "mean_advantage" in stats

    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()


def test_icdar_metric_loss_hybrid():
    processor = DummyProcessor()
    metric_loss = ICDARMetricLoss(
        processor=processor,
        loss_type="hybrid",
        lambda_metric=0.4,
        max_gen_tokens=10,
    )
    model = DummyModel(len(processor.tokenizer))

    inputs = {
        "input_ids": torch.tensor([[7, 8, 2, 4, 2, 5, 2, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 2, 4, 2, 5, 2, 1]]),
    }

    loss, stats = metric_loss(model, inputs)
    assert "sft_ce_loss" in stats
    assert "scst_loss" in stats
    assert "total_loss" in stats

    loss.backward()
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any()


def test_gpu_tensor_reward_module():
    from src.training.icdar_loss import ICDARGPUTensorReward

    tokenizer = DummyTokenizer()
    reward_module = ICDARGPUTensorReward(tokenizer)

    # Identical table sequence: | 1 . 2 |
    tokens_target = torch.tensor([2, 4, 6, 5, 2])
    tokens_pred = torch.tensor([2, 4, 6, 5, 2])

    reward, info = reward_module(tokens_pred, tokens_target)
    assert isinstance(reward, torch.Tensor)
    assert abs(reward.item() - 1.0) < 1e-4
    assert abs(info["r_num"] - 1.0) < 1e-4
    assert abs(info["r_struct"] - 1.0) < 1e-4

    # Divergent sequence: digits missing
    tokens_divergent = torch.tensor([7, 8, 7])
    reward_div, info_div = reward_module(tokens_divergent, tokens_target)
    assert reward_div.item() < reward.item()


def test_float_proximity_weight_curve():
    from src.training.icdar_loss import compute_float_proximity_weight

    # <= 1% error -> almost zero penalty (<= min_near_weight)
    w_exact, err_exact = compute_float_proximity_weight(100.0, 100.0, near_tol=0.01, rel_tol=0.05, min_near_weight=0.05, max_numeric_weight=4.0)
    w_1pct, err_1pct = compute_float_proximity_weight(99.2, 100.0, near_tol=0.01, rel_tol=0.05, min_near_weight=0.05, max_numeric_weight=4.0)
    w_3pct, err_3pct = compute_float_proximity_weight(97.0, 100.0, near_tol=0.01, rel_tol=0.05, min_near_weight=0.05, max_numeric_weight=4.0)
    w_20pct, err_20pct = compute_float_proximity_weight(80.0, 100.0, near_tol=0.01, rel_tol=0.05, min_near_weight=0.05, max_numeric_weight=4.0)
    w_none, _ = compute_float_proximity_weight(None, 100.0, near_tol=0.01, rel_tol=0.05, min_near_weight=0.05, max_numeric_weight=4.0)

    assert w_exact <= w_1pct <= 0.05 + 1e-6
    assert w_1pct < w_3pct < 1.0
    assert 1.0 < w_20pct <= 4.0
    assert w_none == 4.0


def test_cardinality_penalty_points_and_tables():
    from src.training.icdar_loss import compute_cardinality_penalty

    # Same number of points and tables -> multiplier == 1.0
    mult_eq, _ = compute_cardinality_penalty(
        num_pred_points=10, num_target_points=10, num_pred_rows=5, num_target_rows=5, num_pred_tables=2, num_target_tables=2
    )
    assert abs(mult_eq - 1.0) < 1e-6

    # Missing half the points -> multiplier > 1.0
    mult_pts, _ = compute_cardinality_penalty(
        num_pred_points=5, num_target_points=10, num_pred_rows=5, num_target_rows=5, num_pred_tables=1, num_target_tables=1
    )
    assert mult_pts > 1.5

    # Missing second table -> even higher multiplier
    mult_tbl, _ = compute_cardinality_penalty(
        num_pred_points=5, num_target_points=10, num_pred_rows=3, num_target_rows=6, num_pred_tables=1, num_target_tables=2
    )
    assert mult_tbl > mult_pts

    # Column count mismatch -> penalizes structural column divergence
    mult_cols, stats_cols = compute_cardinality_penalty(
        num_pred_points=10, num_target_points=10, num_pred_rows=5, num_target_rows=5,
        num_pred_tables=1, num_target_tables=1, num_pred_cols=2, num_target_cols=4
    )
    assert mult_cols > 1.0
    assert abs(stats_cols["col_rel_diff"] - 0.5) < 1e-6


def test_token_weighted_loss_with_focal_gamma():
    tokenizer = DummyTokenizer()
    loss_module = ICDARTokenWeightedLoss(tokenizer, numeric_weight=3.0, structural_weight=2.0, focal_gamma=1.5)

    logits = torch.randn(2, 6, len(tokenizer), requires_grad=True)
    labels = torch.tensor([
        [-100, -100, 2, 4, 6, 1],
        [-100, 7, 2, 8, 9, 1],
    ])

    loss = loss_module(logits, labels)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_scst_loss_gpu_tensor_reward():
    processor = DummyProcessor()
    scst_module = ICDARSCSTLoss(processor, max_gen_tokens=10, reward_mode="gpu_tensor")
    model = DummyModel(len(processor.tokenizer))

    inputs = {
        "input_ids": torch.tensor([[7, 8, 2, 4, 2, 5, 2, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 2, 4, 2, 5, 2, 1]]),
    }

    loss, stats = scst_module(model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert "mean_r_sampled" in stats
    assert "mean_r_greedy" in stats
    assert "mean_advantage" in stats

    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()

