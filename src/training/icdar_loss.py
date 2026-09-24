"""
Loss functions aligned with official ICDAR 2026 Task 2 metrics (TEDS and RMS).
Supports Self-Critical Sequence Training (SCST / Policy Gradient) and Differentiable Token-Weighted Surrogate Loss.
Optimized for 100% GPU tensor execution to maximize throughput and eliminate CPU bottlenecks.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics.table_metrics import compute_icdar_score
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ICDARGPUTensorReward(nn.Module):
    """GPU-native tensor reward module computing TEDS and RMS proxies directly on CUDA tensors.
    
    Operates 100% on GPU tensors with zero CPU synchronization and zero string decoding:
    - Tensor-RMS: Numerical token set matching and frequency alignment on GPU.
    - Tensor-TEDS: Structural token sequence alignment (unigram + bigram Dice) on GPU.
    """

    def __init__(self, tokenizer: Any, vocab_size: Optional[int] = None):
        super().__init__()
        self.tokenizer = tokenizer
        tok_vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 0
        self.vocab_size = max(tok_vocab_size, vocab_size or 0, 151936)

        is_numeric = torch.zeros(self.vocab_size, dtype=torch.bool)
        is_structural = torch.zeros(self.vocab_size, dtype=torch.bool)

        vocab_dict = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        for token, idx in vocab_dict.items():
            if idx >= self.vocab_size:
                continue
            clean_tok = token.replace("Ġ", "").replace(" ", "").strip()
            if clean_tok and all(c in "0123456789.-" for c in clean_tok):
                is_numeric[idx] = True
            elif clean_tok in ("|", "-", "--", "---", "\n"):
                is_structural[idx] = True

        self.register_buffer("is_numeric", is_numeric)
        self.register_buffer("is_structural", is_structural)

    def forward(
        self,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes composite reward directly on GPU token tensors.
        
        Args:
            pred_tokens: 1D tensor of predicted token IDs on GPU.
            target_tokens: 1D tensor of target token IDs on GPU.
        Returns:
            Tuple of (reward_tensor_scalar, info_dict)
        """
        pred_tokens = pred_tokens.view(-1)
        target_tokens = target_tokens.view(-1)

        # 1. GPU Tensor-RMS (Numerical fidelity proxy)
        num_mask_p = self.is_numeric[pred_tokens]
        num_mask_t = self.is_numeric[target_tokens]
        pred_nums = pred_tokens[num_mask_p]
        target_nums = target_tokens[num_mask_t]

        if target_nums.numel() == 0:
            r_num = torch.tensor(1.0 if pred_nums.numel() == 0 else 0.0, device=pred_tokens.device)
        elif pred_nums.numel() == 0:
            r_num = torch.tensor(0.0, device=pred_tokens.device)
        else:
            c_p = torch.bincount(pred_nums, minlength=self.vocab_size)
            c_t = torch.bincount(target_nums, minlength=self.vocab_size)
            inter_num = torch.minimum(c_p, c_t).sum().float()
            p = inter_num / (pred_nums.numel() + 1e-8)
            r = inter_num / (target_nums.numel() + 1e-8)
            r_num = (2.0 * p * r) / (p + r + 1e-8)

        # 2. GPU Tensor-TEDS (Structural tree fidelity proxy via Markdown skeleton)
        struct_mask_p = self.is_structural[pred_tokens]
        struct_mask_t = self.is_structural[target_tokens]
        pred_struct = pred_tokens[struct_mask_p]
        target_struct = target_tokens[struct_mask_t]

        if target_struct.numel() == 0:
            r_struct = torch.tensor(1.0 if pred_struct.numel() == 0 else 0.0, device=pred_tokens.device)
        elif pred_struct.numel() == 0:
            r_struct = torch.tensor(0.0, device=pred_tokens.device)
        else:
            c_p = torch.bincount(pred_struct, minlength=self.vocab_size)
            c_t = torch.bincount(target_struct, minlength=self.vocab_size)
            inter_1 = torch.minimum(c_p, c_t).sum().float()
            dice_1 = (2.0 * inter_1) / (pred_struct.numel() + target_struct.numel() + 1e-8)

            if pred_struct.numel() > 1 and target_struct.numel() > 1:
                bi_p = pred_struct[:-1] * self.vocab_size + pred_struct[1:]
                bi_t = target_struct[:-1] * self.vocab_size + target_struct[1:]
                inter_2 = torch.isin(bi_p, bi_t).sum().float()
                dice_2 = (2.0 * inter_2) / (bi_p.numel() + bi_t.numel() + 1e-8)
                r_struct = 0.5 * (dice_1 + dice_2)
            else:
                r_struct = dice_1

        total_reward = 0.5 * (r_num + r_struct)
        return total_reward, {
            "r_num": r_num.item(),
            "r_struct": r_struct.item(),
            "reward": total_reward.item(),
        }


class ICDARTokenWeightedLoss(nn.Module):
    """Differentiable surrogate loss weighting structural and numerical table tokens.
    
    Assigns higher penalty to:
    - Numeric tokens (digits, decimal points): proxy for RMS.
    - Structural tokens (pipes, row delimiters): proxy for TEDS.
    Optimized for GPU memory efficiency: slices active table tokens prior to loss computation.
    """

    def __init__(
        self,
        tokenizer: Any,
        numeric_weight: float = 3.0,
        structural_weight: float = 2.0,
        vocab_size: Optional[int] = None,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.numeric_weight = numeric_weight
        self.structural_weight = structural_weight
        self.focal_gamma = focal_gamma
        tok_vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 0
        self.vocab_size = max(tok_vocab_size, vocab_size or 0, 151936)
        self._build_token_weights()

    def _build_token_weights(self):
        """Precomputes weights for special tokens in the tokenizer vocabulary."""
        weights = torch.ones(self.vocab_size, dtype=torch.float32)

        vocab_dict = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else {}
        for token, idx in vocab_dict.items():
            if idx >= self.vocab_size:
                continue
            clean_tok = token.replace("Ġ", "").replace(" ", "").strip()
            if clean_tok and all(c in "0123456789.-" for c in clean_tok):
                weights[idx] = self.numeric_weight
            elif clean_tok in ("|", "-", "--", "---", "\n"):
                weights[idx] = self.structural_weight

        self.register_buffer("token_weights", weights)

    @property
    def weights(self) -> torch.Tensor:
        return self.token_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes weighted cross-entropy loss over target tokens on GPU.
        
        Args:
            logits: (B, L, Vocab)
            labels: (B, L) where non-targets are -100
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        flat_logits = shift_logits.view(-1, shift_logits.shape[-1])[valid_mask.view(-1)]
        flat_labels = shift_labels.view(-1)[valid_mask.view(-1)]

        w = self.token_weights.to(flat_logits.device)
        if w.shape[0] < flat_logits.shape[-1]:
            w = F.pad(w, (0, flat_logits.shape[-1] - w.shape[0]), value=1.0)
        elif w.shape[0] > flat_logits.shape[-1]:
            w = w[: flat_logits.shape[-1]]

        if self.focal_gamma > 0.0:
            ce = F.cross_entropy(flat_logits, flat_labels, weight=w, reduction="none")
            pt = torch.exp(-ce.detach())
            focal_w = (1.0 - pt) ** self.focal_gamma
            loss = (focal_w * ce).mean()
        else:
            loss = F.cross_entropy(flat_logits, flat_labels, weight=w, reduction="mean")

        return loss


class ICDARSCSTLoss(nn.Module):
    """Self-Critical Sequence Training (SCST) loss optimizing exact ICDAR TEDS and RMS.
    
    Generates two predictions per sample:
    1. Sampled sequence y^s (exploration)
    2. Greedy sequence y^g (baseline)
    
    Computes reward R(y) = 0.5 * (TEDS + RMS).
    Advantage A = R(y^s) - R(y^g).
    Loss: L_SCST = - A * mean(log P(y^s)).
    
    Features 'gpu_tensor' reward mode for blazing-fast in-VRAM reward computation without CPU stalls.
    """

    def __init__(
        self,
        processor: Any,
        rel_tol: float = 0.05,
        temperature: float = 0.7,
        max_gen_tokens: int = 512,
        reward_mode: str = "gpu_tensor",
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.processor = processor
        self.rel_tol = rel_tol
        self.temperature = temperature
        self.max_gen_tokens = max_gen_tokens
        self.reward_mode = reward_mode
        self.gpu_reward = ICDARGPUTensorReward(processor.tokenizer, vocab_size=vocab_size)

        # Pre-cache assistant header tokens for GPU sliding-window search
        im_start_fn = getattr(self.processor.tokenizer, "convert_tokens_to_ids", None)
        im_start_id = im_start_fn("<|im_start|>") if im_start_fn else None
        encode_fn = getattr(self.processor.tokenizer, "encode", None)
        assistant_ids = encode_fn("assistant", add_special_tokens=False) if encode_fn else []
        if im_start_id is not None and assistant_ids:
            h = [im_start_id] + assistant_ids + [198]  # with newline '\n'
            self.register_buffer("header_tensor", torch.tensor(h, dtype=torch.long))
        else:
            self.header_tensor = None

    def _find_prompt_end_idx(self, input_ids_row: torch.Tensor, labels_row: Optional[torch.Tensor] = None) -> int:
        """Finds prompt end index without CPU roundtrip whenever possible."""
        if labels_row is not None:
            mask = (labels_row != -100).nonzero(as_tuple=True)[0]
            if mask.numel() > 0 and mask[0].item() > 0:
                return mask[0].item()

        if self.header_tensor is not None and input_ids_row.numel() >= self.header_tensor.numel():
            h_tensor = self.header_tensor.to(input_ids_row.device)
            windows = input_ids_row.unfold(0, h_tensor.numel(), 1)
            matches = (windows == h_tensor).all(dim=-1).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                return (matches[0] + h_tensor.numel()).item()

        return input_ids_row.numel() // 2

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

        prompt_end_idx = self._find_prompt_end_idx(input_ids[0], labels[0])
        if prompt_end_idx <= 0 or prompt_end_idx >= input_ids.shape[1]:
            zero_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return zero_loss, {"r_sampled": 0.0, "r_greedy": 0.0, "advantage": 0.0}

        prompt_input_ids = input_ids[:, :prompt_end_idx]
        prompt_attention_mask = attention_mask[:, :prompt_end_idx]
        prompt_len = prompt_input_ids.shape[1]

        # Prepare visual kwargs safely respecting vision patch flatten layout
        extra_kwargs = {}
        if "pixel_values" in inputs and "image_grid_thw" in inputs:
            grid_thw = inputs["image_grid_thw"]
            patches_per_image = [int(g[0].item() * g[1].item() * g[2].item()) for g in grid_thw]
            start_patch = sum(patches_per_image[:sample_idx])
            num_patches = patches_per_image[sample_idx]
            extra_kwargs["pixel_values"] = inputs["pixel_values"][start_patch : start_patch + num_patches]
            extra_kwargs["image_grid_thw"] = grid_thw[sample_idx : sample_idx + 1]

        for k, v in inputs.items():
            if k in ("labels", "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"):
                continue
            if isinstance(v, torch.Tensor) and v.shape[0] == inputs["input_ids"].shape[0]:
                extra_kwargs[k] = v[sample_idx : sample_idx + 1]
            else:
                extra_kwargs[k] = v

        target_tokens = input_ids[0, prompt_end_idx:]
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        eos_id = self.processor.tokenizer.eos_token_id

        # 1. Generate Greedy Baseline (No grad) & 2. Generate Sampled Output
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
            greedy_tokens = greedy_out[0, prompt_len:]

            if self.reward_mode == "gpu_tensor":
                # Blazing fast in-VRAM reward evaluation
                r_greedy_t, _ = self.gpu_reward(greedy_tokens, target_tokens)
                r_sampled_t, _ = self.gpu_reward(gen_tokens, target_tokens)
                advantage = r_sampled_t - r_greedy_t
                reward_greedy = r_greedy_t.item()
                reward_sampled = r_sampled_t.item()
            else:
                # Exact CPU Zhang-Shasha & Pandas evaluation
                pred_greedy = self.processor.decode(greedy_tokens, skip_special_tokens=True)
                pred_sampled = self.processor.decode(gen_tokens, skip_special_tokens=True)
                target_str = self.processor.decode(target_tokens, skip_special_tokens=True)
                reward_greedy = compute_icdar_score(pred_greedy, target_str, rel_tol=self.rel_tol)["icdar_score"]
                reward_sampled = compute_icdar_score(pred_sampled, target_str, rel_tol=self.rel_tol)["icdar_score"]
                advantage = torch.tensor(reward_sampled - reward_greedy, device=input_ids.device)

        info = {
            "r_sampled": reward_sampled,
            "r_greedy": reward_greedy,
            "advantage": advantage.item() if isinstance(advantage, torch.Tensor) else float(advantage),
        }

        if len(gen_tokens) == 0:
            zero_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return zero_loss, info

        # Forward pass on sampled tokens to compute log probabilities with gradient
        forward_kwargs = dict(extra_kwargs)
        forward_kwargs["input_ids"] = sampled_out
        forward_kwargs["attention_mask"] = torch.ones_like(sampled_out)

        out = model(**forward_kwargs)
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
        reward_mode: str = "gpu_tensor",
        focal_gamma: float = 0.0,
        vocab_size: Optional[int] = None,
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
                reward_mode=reward_mode,
                vocab_size=vocab_size,
            )
        else:
            self.scst_module = None

        if loss_type in ("token_weighted",):
            self.weighted_module = ICDARTokenWeightedLoss(
                tokenizer=processor.tokenizer,
                numeric_weight=numeric_weight,
                structural_weight=structural_weight,
                vocab_size=vocab_size,
                focal_gamma=focal_gamma,
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
            if sft_loss is not None:
                total_loss = sft_loss
            else:
                out = model(**inputs)
                if out.logits is not None:
                    total_loss = self.weighted_module(out.logits, inputs["labels"])
                else:
                    total_loss = out.loss
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
