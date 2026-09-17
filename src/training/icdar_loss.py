"""
Loss functions aligned with official ICDAR 2026 Task 2 metrics (TEDS and RMS).
Supports Self-Critical Sequence Training (SCST / Policy Gradient) and Differentiable Token-Weighted Surrogate Loss.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics.table_metrics import compute_icdar_score
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ICDARTokenWeightedLoss(nn.Module):
    """Differentiable surrogate loss weighting structural and numerical table tokens.
    
    Assigns higher penalty to:
    - Numeric tokens (digits, decimal points): proxy for RMS.
    - Structural tokens (pipes, row delimiters): proxy for TEDS.
    """

    def __init__(
        self,
        tokenizer: Any,
        numeric_weight: float = 3.0,
        structural_weight: float = 2.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.numeric_weight = numeric_weight
        self.structural_weight = structural_weight
        self._build_token_weights()

    def _build_token_weights(self):
        """Precomputes weights for special tokens in the tokenizer vocabulary."""
        vocab_size = len(self.tokenizer)
        weights = torch.ones(vocab_size, dtype=torch.float32)

        for token, idx in self.tokenizer.get_vocab().items():
            # Check if numeric (digits 0-9, decimal point, negative sign)
            clean_tok = token.replace("Ġ", "").replace(" ", "").strip()
            if clean_tok and all(c in "0123456789.-" for c in clean_tok):
                weights[idx] = self.numeric_weight
            elif clean_tok in ("|", "-", "--", "---", "\n"):
                weights[idx] = self.structural_weight

        self.register_buffer("token_weights", weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes weighted cross-entropy loss over target tokens.
        
        Args:
            logits: (B, L, Vocab)
            labels: (B, L) where non-targets are -100
        """
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        target_ids = torch.where(valid_mask, shift_labels, torch.zeros_like(shift_labels))
        weights = self.token_weights[target_ids]
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_loss = -log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        weighted_loss = (per_token_loss * weights).sum() / (weights.sum() + 1e-8)

        return weighted_loss


class ICDARSCSTLoss(nn.Module):
    """Self-Critical Sequence Training (SCST) loss optimizing exact ICDAR TEDS and RMS.
    
    Generates two predictions per sample:
    1. Sampled sequence y^s (exploration)
    2. Greedy sequence y^g (baseline)
    
    Computes reward R(y) = 0.5 * (TEDS + RMS).
    Advantage A = R(y^s) - R(y^g).
    Loss: L_SCST = - A * mean(log P(y^s)).
    """

    def __init__(
        self,
        processor: Any,
        rel_tol: float = 0.05,
        temperature: float = 0.7,
        max_gen_tokens: int = 512,
    ):
        super().__init__()
        self.processor = processor
        self.rel_tol = rel_tol
        self.temperature = temperature
        self.max_gen_tokens = max_gen_tokens

    def compute_scst_sample_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        sample_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes SCST policy gradient loss for a single sample in the batch."""
        input_ids = inputs["input_ids"][sample_idx : sample_idx + 1]
        attention_mask = inputs["attention_mask"][sample_idx : sample_idx + 1]
        labels = inputs["labels"][sample_idx : sample_idx + 1]

        # Identify prompt boundary (where labels != -100 begins)
        target_mask = labels[0] != -100
        if not target_mask.any():
            zero_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return zero_loss, {"r_sampled": 0.0, "r_greedy": 0.0, "advantage": 0.0}

        prompt_end_idx = (target_mask == True).nonzero(as_tuple=True)[0][0].item()
        prompt_input_ids = input_ids[:, :prompt_end_idx]
        prompt_attention_mask = attention_mask[:, :prompt_end_idx]
        prompt_len = prompt_input_ids.shape[1]

        # Prepare visual kwargs
        extra_kwargs = {
            k: v[sample_idx : sample_idx + 1]
            for k, v in inputs.items()
            if k not in ("labels", "input_ids", "attention_mask")
        }

        # Decode ground truth table string
        target_tokens = input_ids[0, prompt_end_idx:]
        target_str = self.processor.decode(target_tokens, skip_special_tokens=True)

        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        eos_id = self.processor.tokenizer.eos_token_id

        # 1. Generate Greedy Baseline (No grad)
        with torch.no_grad():
            greedy_out = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.max_gen_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                **extra_kwargs,
            )
            pred_greedy = self.processor.decode(greedy_out[0, prompt_len:], skip_special_tokens=True)
            reward_greedy = compute_icdar_score(pred_greedy, target_str, rel_tol=self.rel_tol)["icdar_score"]

            # 2. Generate Sampled Output (Exploration)
            sampled_out = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.max_gen_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                **extra_kwargs,
            )
            gen_tokens = sampled_out[0, prompt_len:]
            pred_sampled = self.processor.decode(gen_tokens, skip_special_tokens=True)
            reward_sampled = compute_icdar_score(pred_sampled, target_str, rel_tol=self.rel_tol)["icdar_score"]

        advantage = reward_sampled - reward_greedy
        info = {
            "r_sampled": reward_sampled,
            "r_greedy": reward_greedy,
            "advantage": advantage,
        }

        if len(gen_tokens) == 0:
            zero_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return zero_loss, info

        # Forward pass on sampled tokens to compute log probabilities with gradient
        forward_kwargs = dict(extra_kwargs)
        forward_kwargs["input_ids"] = sampled_out
        forward_kwargs["attention_mask"] = torch.ones_like(sampled_out)

        out = model(**forward_kwargs)
        # Shift logits corresponding to generated tokens
        logits = out.logits[0, prompt_len - 1 : -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=gen_tokens.unsqueeze(-1)).squeeze(-1)
        seq_log_prob = token_log_probs.mean()

        # Policy Gradient loss: - advantage * log_prob
        loss_scst = - advantage * seq_log_prob

        return loss_scst, info

    def forward(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes SCST loss averaged over batch."""
        batch_size = inputs["input_ids"].shape[0]
        losses = []
        sampled_rewards = []
        greedy_rewards = []
        advantages = []

        for b in range(batch_size):
            loss_b, info_b = self.compute_scst_sample_loss(model, inputs, sample_idx=b)
            losses.append(loss_b)
            sampled_rewards.append(info_b["r_sampled"])
            greedy_rewards.append(info_b["r_greedy"])
            advantages.append(info_b["advantage"])

        mean_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True)
        stats = {
            "mean_r_sampled": sum(sampled_rewards) / len(sampled_rewards) if sampled_rewards else 0.0,
            "mean_r_greedy": sum(greedy_rewards) / len(greedy_rewards) if greedy_rewards else 0.0,
            "mean_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
        }
        return mean_loss, stats


class ICDARMetricLoss(nn.Module):
    """Hybrid loss module combining SFT Cross-Entropy with ICDAR Metric Loss.
    
    Loss = (1 - lambda_metric) * L_SFT + lambda_metric * L_SCST
    """

    def __init__(
        self,
        processor: Any,
        loss_type: str = "scst",  # 'scst', 'token_weighted', 'hybrid'
        lambda_metric: float = 0.3,
        rel_tol: float = 0.05,
        temperature: float = 0.7,
        max_gen_tokens: int = 512,
        numeric_weight: float = 3.0,
        structural_weight: float = 2.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_metric = lambda_metric
        self.processor = processor

        if loss_type in ("scst", "hybrid"):
            self.scst_module = ICDARSCSTLoss(
                processor=processor,
                rel_tol=rel_tol,
                temperature=temperature,
                max_gen_tokens=max_gen_tokens,
            )
        else:
            self.scst_module = None

        if loss_type in ("token_weighted",):
            self.weighted_module = ICDARTokenWeightedLoss(
                tokenizer=processor.tokenizer,
                numeric_weight=numeric_weight,
                structural_weight=structural_weight,
            )
        else:
            self.weighted_module = None

    def forward(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        sft_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the final combined loss."""
        metrics_info: Dict[str, float] = {}

        if self.loss_type == "token_weighted":
            out = model(**inputs)
            total_loss = self.weighted_module(out.logits, inputs["labels"])
            metrics_info["token_weighted_loss"] = total_loss.item()
            return total_loss, metrics_info

        # SCST or Hybrid
        if sft_loss is None:
            out = model(**inputs)
            sft_loss = out.loss

        metrics_info["sft_ce_loss"] = sft_loss.item()

        if self.lambda_metric <= 0.0 or self.scst_module is None:
            return sft_loss, metrics_info

        scst_loss, scst_stats = self.scst_module(model, inputs)
        metrics_info.update(scst_stats)
        metrics_info["scst_loss"] = scst_loss.item()

        total_loss = (1.0 - self.lambda_metric) * sft_loss + self.lambda_metric * scst_loss
        metrics_info["total_loss"] = total_loss.item()

        return total_loss, metrics_info
