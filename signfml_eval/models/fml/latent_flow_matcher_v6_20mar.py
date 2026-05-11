#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Flow Matcher V6 (20 Mar) — Iterative Refinement Sampling
================================================================
Author: SignFML Research
Date: 20 Mar 2026

Based on V5 (28 Feb). Architecture and training identical.
Changes from V5:
    - Added sample_with_refinement(): SDEdit-style iterative refinement at inference.
      Reduces DTW-MJE by re-denoising the sample multiple times from partial noise.
      No new trainable parameters — only affects inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from transformers import MBartModel

# Import V2 components
try:
    from .flow_matching_v5_shared import (
        FlowMatchingBlockV2,
        FlowMatchingScheduler,
        SafeFlowMatchingLoss
    )
    from .transformer_prior_v5_28feb import TransformerPriorV2
except ImportError:
    from flow_matching_v5_shared import (
        FlowMatchingBlockV2,
        FlowMatchingScheduler,
        SafeFlowMatchingLoss
    )
    from transformer_prior_v5_28feb import TransformerPriorV2


class TransformerLengthPredictor(nn.Module):
    """Predicts sequence length using a small Transformer Encoder.
    
    Better at capturing linguistic dependencies compared to a simple MLP.
    Output is in LOG-SPACE: pred ≈ log(num_frames).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2, 
                 num_heads: int = 4, dropout: float = 0.2, init_mean_length: float = 120.0):
        super().__init__()
        
        # Projection to internal hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mini Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling (same as V4 but after transformer)
        self.attn_query = nn.Linear(hidden_dim, 1)
        
        # Output Head (No Softplus here as Sigmoid handles range in forward)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.init_mean_length = init_mean_length
        self.sigmoid_offset = 4.0 # Center of sigmoid scaling
        self._reset_bias()

    def _reset_bias(self):
        import math
        # target_log: want output of forward to be log(init_mean_length)
        target_log = math.log(max(self.init_mean_length, 1.0)) # ~4.79
        
        # Invert the sigmoid scaling: x = offset + logit((target - min) / (max - min))
        min_log, max_log = 1.0, 6.5
        ratio = (target_log - min_log) / (max_log - min_log)
        ratio = max(min(ratio, 0.999), 0.001)
        
        # logit(p) = log(p/(1-p))
        pre_sigmoid = self.sigmoid_offset + math.log(ratio / (1 - ratio))
        
        with torch.no_grad():
            self.head[-1].bias.fill_(pre_sigmoid)

    def forward(self, text_features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # text_features: [B, T, D]
        x = self.input_proj(text_features) # [B, T, H]
        
        # Transformer processing
        if mask is not None:
            # Transformer mask is [B, T] where True is MASKED
            # Our input mask is [B, T] where True is VALID
            src_key_padding_mask = ~mask 
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)
            
        # Attention Pooling
        attn_scores = self.attn_query(x).squeeze(-1) # [B, T]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1) # [B, H]
        
        # Prediction
        pred_log_length = self.head(pooled).squeeze(-1) # [B]
        
        # Soft-clamp to reasonable range (log(10) to log(512)) -> (~2.3 to 6.2)
        min_log, max_log = 1.0, 6.5
        pred_log_length = min_log + (max_log - min_log) * torch.sigmoid(pred_log_length - self.sigmoid_offset)
        
        return pred_log_length


class MaskProjector(nn.Module):
    """Projects binary masks [B, T, 1] to hidden dimension [B, T, D]."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, hidden_dim, bias=False)
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        return self.proj(mask.float())


class LatentFlowMatcherV6(nn.Module):
    """
    V5: mBART-50 Large + Transformer Length Predictor.
    Optimized for SCST (Self-Critical) training.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        text_encoder_name: str = 'facebook/mbart-large-50-many-to-many-mmt',
        hidden_dim: int = 512,
        num_flow_layers: int = 6,
        num_prior_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_ssm_prior: bool = True,
        use_sync_guidance: bool = False,
        cfg_dropout_rate: float = 0.1,
        pred_length_mask_ratio: float = 0.0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_ssm_prior = use_ssm_prior
        self.cfg_dropout_rate = cfg_dropout_rate
        self.pred_length_mask_ratio = pred_length_mask_ratio
        
        # mBART-50 encoder
        print(f"🆕 V5: Loading mBART-50 encoder from: {text_encoder_name}")
        full_mbart = MBartModel.from_pretrained(text_encoder_name)
        self.text_encoder = full_mbart.encoder
        del full_mbart
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        text_dim = self.text_encoder.config.hidden_size
        print(f"  📐 Text encoder dim: {text_dim} (mBART-50)")
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # V5 CHANGE: Transformer-based Length Predictor
        self.length_predictor = TransformerLengthPredictor(
            input_dim=text_dim, 
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=dropout
        )
        self.length_loss_fn = nn.SmoothL1Loss()
        
        self.scheduler = FlowMatchingScheduler()
        self.flow_block = FlowMatchingBlockV2(
            data_dim=latent_dim,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        if use_ssm_prior:
            self.ssm_prior = TransformerPriorV2(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_prior_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
        else:
            self.ssm_prior = None
        
        self.flow_loss_fn = SafeFlowMatchingLoss(max_loss=1000.0, use_huber=True, huber_delta=5.0)

    def encode_text(self, text_tokens: torch.Tensor,
                    attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes text tokens into projected features."""
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=text_tokens, attention_mask=attention_mask)
            text_encoder_feats = outputs.last_hidden_state
            # Zero-out padding positions so fresh features match cached features
            # (cache trims to actual_len then re-pads with zeros)
            text_encoder_feats = text_encoder_feats * attention_mask.unsqueeze(-1).float()

        text_feats = self.text_proj(text_encoder_feats)
        return text_feats, attention_mask, text_encoder_feats

    def get_condition(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batch to get text condition.

        If batch contains 'raw_text_feats' (precomputed mBART cache), skips the
        expensive mBART forward pass and only runs the lightweight text_proj linear layer.
        """
        if 'raw_text_feats' in batch:
            raw_text_feats = batch['raw_text_feats']  # [B, T_text, 1024] — already on device
            text_mask = batch['attention_mask']
            text_feats = self.text_proj(raw_text_feats)  # [B, T_text, hidden_dim]
            return text_feats, text_mask, raw_text_feats
        text_feats, text_mask, raw_text_feats = self.encode_text(batch['text_tokens'], batch['attention_mask'])
        return text_feats, text_mask, raw_text_feats

    def forward(self, batch: Dict[str, torch.Tensor], 
                gt_latent: torch.Tensor,
                condition: Optional[torch.Tensor] = None, 
                text_mask: Optional[torch.Tensor] = None,
                raw_text_feats: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute flow matching and length prediction losses."""
        # 1. Condition
        if condition is None:
            condition, text_mask, raw_text_feats = self.get_condition(batch)
        B, T_text, _ = condition.shape
        D = self.latent_dim
        
        # 2. Length Prediction Loss
        gt_lengths = batch['seq_lengths'].float()
        gt_log_lengths = torch.log(gt_lengths + 1e-6)
        
        # Apply robustness masking for length prediction (V4 style)
        lp_input = raw_text_feats
        if self.training and self.pred_length_mask_ratio > 0:
            mask_indices = torch.rand(raw_text_feats.shape[:2], device=raw_text_feats.device) < self.pred_length_mask_ratio
            lp_input = raw_text_feats.clone()
            lp_input[mask_indices] = 0.0
            
        pred_log_lengths = self.length_predictor(lp_input, text_mask.bool())
        loss_length = self.length_loss_fn(pred_log_lengths, gt_log_lengths)
        
        # 3. Flow Matching Loss
        x1 = gt_latent # Target latents [B, T_pose, D]
        T_pose = x1.shape[1]
        valid_mask = torch.arange(T_pose, device=x1.device)[None, :] < batch['seq_lengths'][:, None]
        
        # Add Noise (Standard Flow Matching)
        ts = torch.rand(B, device=x1.device)
        xt, v_gt, _ = self.scheduler.add_noise(x1, ts)
        
        # SSM Prior
        loss_prior = torch.tensor(0.0, device=x1.device)
        v_prior = None
        if self.ssm_prior is not None:
            # Correct call to TransformerPriorV2(z, t, text, pose_mask, text_mask)
            v_prior = self.ssm_prior(xt, ts, condition, pose_mask=valid_mask, text_mask=text_mask)
            # Prior recon loss (MSE on velocity)
            prior_diff = (v_prior - v_gt.detach()) ** 2
            loss_prior = (prior_diff * valid_mask.unsqueeze(-1).float()).sum() / (valid_mask.sum() * D).clamp(min=1.0)
        
        # Main Flow Matching
        v_flow = self.flow_block(xt, ts, condition, x_mask=valid_mask, condition_mask=text_mask)
        v_flow = torch.nan_to_num(v_flow, nan=0.0)

        # 🆕 Prior guidance during training (consistent with inference):
        # Inference: v_total = v_flow + lambda_t * v_prior
        # → Flow block learns residual: v_target = v_gt - lambda_t * v_prior
        if v_prior is not None:
            lambda_t = 0.5 * (1.0 - ts).view(-1, 1, 1)  # anneal like inference
            v_target = v_gt - lambda_t * v_prior.detach()
        else:
            v_target = v_gt
        loss_flow = self.flow_loss_fn(v_flow, v_target, mask=valid_mask)
            
        # Compute est_x1 from the SAME forward pass (for perceptual/pose/velocity losses)
        # est_x1 = xt + (1-t) * v_total, where v_total includes prior guidance
        v_total = v_flow
        if v_prior is not None:
            lambda_t_inf = 0.5 * (1.0 - ts).view(-1, 1, 1)
            v_total = v_flow + lambda_t_inf * v_prior.detach()
        est_x1 = xt + (1.0 - ts.view(-1, 1, 1)) * v_total

        return {
            'loss_flow': loss_flow,
            'loss_length': loss_length,
            'loss_prior': loss_prior,
            'est_x1': est_x1,       # [B, T, D] estimated clean latent
            'timesteps': ts,         # [B] timesteps used
            'valid_mask': valid_mask, # [B, T]
        }

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        condition: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        raw_text_feats: Optional[torch.Tensor] = None,
        steps: int = 50,
        cfg_scale: float = 1.0,
        temperature: float = 1.0,
        target_length: Optional[torch.Tensor] = None,
        return_length: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """Sample from the flow matcher. Returns (sample, x0) or (sample, x0, target_length) if return_length=True."""
        if condition is None:
            condition, text_mask, raw_text_feats = self.get_condition(batch)
        B = condition.shape[0]
        
        # Predict Length if not provided
        if target_length is None:
            pred_log_len = self.length_predictor(raw_text_feats, text_mask.bool())
            target_length = torch.exp(pred_log_len).round().long()
            target_length = torch.clamp(target_length, min=10, max=512)
        
        max_len = target_length.max().item()
        
        # CFG Handling
        # Note: In V5 we simplify CFG to avoid redundant forward passes during sampling

        valid_mask = torch.arange(max_len, device=condition.device)[None, :] < target_length[:, None]
        if temperature <= 0.0:
            # Avoid all-zero initialization collapse at temp=0.0 by using a
            # deterministic condition-derived seed (no new trainable params).
            if text_mask is not None:
                text_mask_f = text_mask.float().unsqueeze(-1)
                pooled_cond = (condition * text_mask_f).sum(dim=1) / text_mask_f.sum(dim=1).clamp(min=1.0)
            else:
                pooled_cond = condition.mean(dim=1)

            cond_seed = torch.tanh(pooled_cond[:, :self.latent_dim])
            x0 = 0.01 * cond_seed.unsqueeze(1).expand(B, max_len, self.latent_dim).contiguous()
        else:
            x0 = torch.randn(B, max_len, self.latent_dim, device=condition.device) * temperature
        x = x0.clone()

        # ODE Integration (Euler)
        dt = 1.0 / steps
        # Pre-compute null condition for CFG (once, not every step)
        null_cond = None
        if cfg_scale != 1.0:
            null_cond = torch.zeros_like(condition)
            null_text_mask = torch.zeros(B, condition.shape[1], device=condition.device, dtype=text_mask.dtype)
            null_text_mask[:, 0] = True  # Keep 1 token to avoid all-masked NaN

        for i in range(steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=x.device)

            # 1. Flow Velocity
            v = self.flow_block(x, t, condition, x_mask=valid_mask, condition_mask=text_mask)

            # 2. CFG (Optional)
            if cfg_scale != 1.0:
                v_uncond = self.flow_block(x, t, null_cond, x_mask=valid_mask, condition_mask=null_text_mask)
                v = v_uncond + cfg_scale * (v - v_uncond)

            # 3. Prior Guidance (V4 style)
            if self.ssm_prior is not None:
                # Correct call: self.ssm_prior(z, t, text, pose_mask, text_mask)
                v_prior = self.ssm_prior(x, t, condition, pose_mask=valid_mask, text_mask=text_mask)
                lambda_t = 0.5 * (1.0 - t_val) # Annealing lambda
                v = v + lambda_t * v_prior
            v = torch.nan_to_num(v, nan=0.0) # Force zero on NaNs
            v = torch.clamp(v, -10.0, 10.0) # Clamp velocity
            x = x + v * dt
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.clamp(x, -20.0, 20.0) # Clamp latents

            if torch.isnan(x).any():
                print(f"  ⚠️ NaN triggered in sample step {i}!")
                break
                
        if return_length:
            return x * valid_mask.unsqueeze(-1), x0, target_length
        return x * valid_mask.unsqueeze(-1), x0

    @torch.no_grad()
    def sample_with_refinement(
        self,
        batch: Dict[str, torch.Tensor],
        condition: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        raw_text_feats: Optional[torch.Tensor] = None,
        steps: int = 50,
        cfg_scale: float = 1.0,
        temperature: float = 1.0,
        target_length: Optional[torch.Tensor] = None,
        return_length: bool = False,
        num_refinements: int = 3,
        refine_start_t: float = 0.3,
    ) -> Tuple[torch.Tensor, ...]:
        """Sample with iterative refinement (SDEdit-style).

        After the initial ODE solve, adds partial noise back to the result
        and re-solves. Each pass refines pose accuracy without changing length.
        No new trainable parameters — uses the same flow_block and prior.

        Args:
            num_refinements: Number of refinement passes after initial sample (default 3)
            refine_start_t: Noise level to restart from (0.3 = 30% noise). Lower = stronger refinement.
        """
        if condition is None:
            condition, text_mask, raw_text_feats = self.get_condition(batch)
        B = condition.shape[0]

        # Predict length once (shared across all refinement passes)
        if target_length is None:
            pred_log_len = self.length_predictor(raw_text_feats, text_mask.bool())
            target_length = torch.exp(pred_log_len).round().long()
            target_length = torch.clamp(target_length, min=10, max=512)

        max_len = target_length.max().item()
        valid_mask = torch.arange(max_len, device=condition.device)[None, :] < target_length[:, None]

        # Pre-compute null condition for CFG (once)
        null_cond = None
        null_text_mask = None
        if cfg_scale != 1.0:
            null_cond = torch.zeros_like(condition)
            null_text_mask = torch.zeros(B, condition.shape[1], device=condition.device, dtype=text_mask.dtype)
            null_text_mask[:, 0] = True

        def _ode_solve(x_start, t_start, t_end, num_steps):
            """Run Euler ODE integration from t_start to t_end."""
            x = x_start
            dt = (t_end - t_start) / num_steps
            for i in range(num_steps):
                t_val = t_start + i * dt
                t = torch.full((B,), t_val, device=x.device)

                v = self.flow_block(x, t, condition, x_mask=valid_mask, condition_mask=text_mask)

                if cfg_scale != 1.0:
                    v_uncond = self.flow_block(x, t, null_cond, x_mask=valid_mask, condition_mask=null_text_mask)
                    v = v_uncond + cfg_scale * (v - v_uncond)

                if self.ssm_prior is not None:
                    v_prior = self.ssm_prior(x, t, condition, pose_mask=valid_mask, text_mask=text_mask)
                    lambda_t = 0.5 * (1.0 - t_val)
                    v = v + lambda_t * v_prior

                v = torch.nan_to_num(v, nan=0.0)
                v = torch.clamp(v, -10.0, 10.0)
                x = x + v * dt
                x = torch.nan_to_num(x, nan=0.0)
                x = torch.clamp(x, -20.0, 20.0)
            return x

        # === Pass 1: Full ODE solve from noise (t=0 → 1) ===
        if temperature <= 0.0:
            if text_mask is not None:
                text_mask_f = text_mask.float().unsqueeze(-1)
                pooled_cond = (condition * text_mask_f).sum(dim=1) / text_mask_f.sum(dim=1).clamp(min=1.0)
            else:
                pooled_cond = condition.mean(dim=1)
            cond_seed = torch.tanh(pooled_cond[:, :self.latent_dim])
            x0 = 0.01 * cond_seed.unsqueeze(1).expand(B, max_len, self.latent_dim).contiguous()
        else:
            x0 = torch.randn(B, max_len, self.latent_dim, device=condition.device) * temperature

        x = _ode_solve(x0, 0.0, 1.0, steps)

        # === Refinement passes: partial noise → re-solve ===
        refine_steps = max(1, int(steps * (1.0 - refine_start_t)))
        for k in range(num_refinements):
            # Flow matching interpolation: x_t = (1-t)*noise + t*x1
            # At t=refine_start_t, mix current result with fresh noise
            noise = torch.randn_like(x) * (temperature if temperature > 0 else 1.0)
            x_noisy = (1.0 - refine_start_t) * noise + refine_start_t * x
            x = _ode_solve(x_noisy, refine_start_t, 1.0, refine_steps)

        x = x * valid_mask.unsqueeze(-1)
        if return_length:
            return x, x0, target_length
        return x, x0

    def get_log_prob(self, latents: torch.Tensor, batch: Dict[str, torch.Tensor],
                     x0: Optional[torch.Tensor] = None,
                     condition: Optional[torch.Tensor] = None,
                     K: int = 4,
                     pred_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Heuristic log-prob for SCST (Multi-step MSE proxy to reduce variance).

        Args:
            pred_lengths: Predicted sequence lengths from model.sample(). If provided,
                          used for valid_mask instead of GT batch['seq_lengths'].
        """
        if condition is None:
            condition, text_mask, _ = self.get_condition(batch)
        else:
            text_mask = batch['attention_mask'].bool()

        B, T_pose, D = latents.shape
        device = latents.device
        # Use predicted lengths if available — they match the actual generated sequence
        seq_lens = pred_lengths if pred_lengths is not None else batch['seq_lengths']
        valid_mask = torch.arange(T_pose, device=device)[None, :] < seq_lens[:, None]
        
        if x0 is None:
            x0 = torch.randn_like(latents) 
            
        # Stratified sampling of t to reduce variance
        all_mse = []
        for k in range(K):
            t_min, t_max = k / K, (k + 1) / K
            ts = t_min + (t_max - t_min) * torch.rand(B, device=device)
            
            xt = (1 - ts[:, None, None]) * x0 + ts[:, None, None] * latents
            vt_gt = latents - x0
            vt_pred = self.flow_block(xt, ts, condition, x_mask=valid_mask, condition_mask=text_mask)
            vt_pred = torch.clamp(vt_pred, -10.0, 10.0) # Safety
            
            # MSE only on valid positions
            mse = ((vt_pred - vt_gt) ** 2) * valid_mask.unsqueeze(-1).float()
            mse = mse.sum(dim=(1, 2)) / (valid_mask.sum(dim=1) * D).clamp(min=1.0)
            all_mse.append(mse)
            
        # Average MSE across K steps
        avg_mse = torch.stack(all_mse, dim=0).mean(dim=0)
        
        return -avg_mse
