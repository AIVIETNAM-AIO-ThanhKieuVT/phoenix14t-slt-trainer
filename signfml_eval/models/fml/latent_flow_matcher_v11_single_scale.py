#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Flow Matcher V11 - Single Scale Ablation
==============================================
This version removes the Multi-Scale (U-Net style) hierarchy and replaces it 
with a simple stack of 9 TemporalBiasedTransformerLayers to maintain 
comparable parameter count and depth.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from transformers import MBartModel

# Reuse V11 components
from .flow_matching_v5_shared import (
    RotaryPositionEmbedding,
    SinusoidalPositionalEmbedding,
    AdaLNZero,
    SafeTimeEmbedding,
    FlowMatchingScheduler,
    SafeFlowMatchingLoss
)
from .transformer_prior_v5_28feb import TransformerPriorV2
from .latent_flow_matcher_v11_24mar import (
    TemporalBiasedTransformerLayer,
    DurationAwareLengthPredictor
)

class SingleScaleFlowBlock(nn.Module):
    """Simple stack of TemporalBiasedTransformerLayers (No Downsampling)."""
    def __init__(
        self,
        data_dim: int = 256,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 9, # 3+3+3 from V11
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        layer_scale_init: float = 0.1,
        local_window: int = 64
    ):
        super().__init__()
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.time_embed = SafeTimeEmbedding(hidden_dim, max_value=5.0)
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        self.text_pos_embed = SinusoidalPositionalEmbedding(dim=hidden_dim, max_seq_len=max_seq_len)
        self.length_embed = nn.Embedding(max_seq_len + 1, hidden_dim)
        self.rope = RotaryPositionEmbedding(dim=hidden_dim // num_heads, max_seq_len=max_seq_len)

        self.layers = nn.ModuleList([
            TemporalBiasedTransformerLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                ffn_dim=hidden_dim * 4, dropout=dropout,
                layer_scale_init=layer_scale_init,
                max_seq_len=max_seq_len, local_window=local_window
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        
        # Zero init output
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, condition, x_mask=None, condition_mask=None, seq_length=None):
        B, T, _ = x.shape
        h = self.input_proj(torch.clamp(x, -50, 50))
        cond = self.cond_proj(torch.clamp(condition, -50, 50))
        cond = self.text_pos_embed(cond)
        t_emb = self.time_embed(t)
        
        if seq_length is not None:
            t_emb = t_emb + self.length_embed(seq_length.clamp(0, 511).long())

        for layer in self.layers:
            h = layer(h, t_emb, cond, rope=self.rope, pose_mask=x_mask, text_mask=condition_mask)
            
        v = self.output_proj(self.final_norm(h))
        return torch.clamp(v, -50, 50)

class LatentFlowMatcherV11_SingleScale(nn.Module):
    """V11 Ablation: Only 1 scale, no hierarchy."""
    def __init__(self, latent_dim=256, text_encoder_name='facebook/mbart-large-50-many-to-many-mmt',
                 hidden_dim=512, num_layers=9, num_heads=8, dropout=0.1, max_seq_len=512,
                 use_ssm_prior=True, local_window=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # MBART
        full_mbart = MBartModel.from_pretrained(text_encoder_name)
        self.text_encoder = full_mbart.encoder
        for p in self.text_encoder.parameters(): p.requires_grad = False
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.length_predictor = DurationAwareLengthPredictor(input_dim=text_dim, hidden_dim=hidden_dim)
        self.length_loss_fn = nn.SmoothL1Loss()
        
        self.scheduler = FlowMatchingScheduler()
        self.flow_block = SingleScaleFlowBlock(
            data_dim=latent_dim, condition_dim=hidden_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
            max_seq_len=max_seq_len, local_window=local_window
        )
        
        self.ssm_prior = TransformerPriorV2(latent_dim=latent_dim, hidden_dim=hidden_dim) if use_ssm_prior else None
        self.flow_loss_fn = SafeFlowMatchingLoss()

    def get_condition(self, batch):
        if 'raw_text_feats' in batch:
            raw = batch['raw_text_feats']
            return self.text_proj(raw), batch['attention_mask'], raw
        # (Simplified for ablation, usually cached)
        return None, None, None 

    def forward(self, batch, gt_latent, condition=None, text_mask=None, raw_text_feats=None):
        if condition is None: condition, text_mask, raw_text_feats = self.get_condition(batch)
        B, T_text, _ = condition.shape
        
        # Length
        gt_len = batch['seq_lengths'].float()
        pred_log_len = self.length_predictor(raw_text_feats, text_mask.bool())
        loss_length = self.length_loss_fn(pred_log_len, torch.log(gt_len + 1e-6))
        
        # Flow
        x1 = gt_latent
        T_pose = x1.shape[1]
        valid_mask = torch.arange(T_pose, device=x1.device)[None, :] < batch['seq_lengths'][:, None]
        
        ts = torch.rand(B, device=x1.device)
        xt, v_gt, _ = self.scheduler.add_noise(x1, ts)
        
        v_prior = self.ssm_prior(xt, ts, condition, pose_mask=valid_mask, text_mask=text_mask) if self.ssm_prior else None
        
        v_flow = self.flow_block(xt, ts, condition, x_mask=valid_mask, condition_mask=text_mask)
        
        if v_prior is not None:
            v_target = v_gt - 0.5 * (1.0 - ts).view(-1, 1, 1) * v_prior.detach()
        else:
            v_target = v_gt
            
        loss_flow = self.flow_loss_fn(v_flow, v_target, mask=valid_mask)
        
        # Estimate x1 for other losses
        v_total = v_flow + (0.5 * (1.0 - ts).view(-1, 1, 1) * v_prior.detach() if v_prior is not None else 0)
        est_x1 = xt + (1.0 - ts.view(-1, 1, 1)) * v_total

        return {
            'loss_flow': loss_flow,
            'loss_length': loss_length,
            'est_x1': est_x1,
            'valid_mask': valid_mask
        }

    @torch.no_grad()
    def sample(self, batch, condition=None, text_mask=None, raw_text_feats=None, 
               steps=20, cfg_scale=1.0, temperature=1.0, target_length=None, 
               return_length=False, ode_method='euler', use_dynamic_cfg=False):
        """Advanced sampling with CFG and various ODE solvers (matching V11)."""
        if condition is None: condition, text_mask, raw_text_feats = self.get_condition(batch)
        if target_length is None:
            target_length = torch.exp(self.length_predictor(raw_text_feats, text_mask.bool())).round().long().clamp(10, 512)
        
        B, max_len = condition.shape[0], target_length.max().item()
        valid_mask = torch.arange(max_len, device=condition.device)[None, :] < target_length[:, None]
        
        # Initial noise
        x = torch.randn(B, max_len, self.latent_dim, device=condition.device) * temperature
        x0 = x.clone()
        
        # Prepare Null Condition for CFG
        null_condition = torch.zeros_like(condition)
        
        def get_velocity(xt, t_val):
            # Conditional velocity
            v_cond = self.flow_block(xt, t_val, condition, x_mask=valid_mask, condition_mask=text_mask)
            if self.ssm_prior:
                v_cond = v_cond + 0.5 * (1.0 - t_val.view(-1, 1, 1)) * self.ssm_prior(xt, t_val, condition, pose_mask=valid_mask, text_mask=text_mask)
            
            if cfg_scale <= 1.0:
                return v_cond
            
            # Unconditional velocity
            v_uncond = self.flow_block(xt, t_val, null_condition, x_mask=valid_mask, condition_mask=text_mask)
            
            # Apply (Dynamic) CFG
            w = cfg_scale
            if use_dynamic_cfg: # Gradually reduce guidance as t -> 1
                w = cfg_scale * (1.0 - t_val[0].item())**0.5
                
            v_final = v_uncond + w * (v_cond - v_uncond)
            return v_final

        dt = 1.0 / steps
        for i in range(steps):
            t_curr = torch.full((B,), i * dt, device=x.device)
            
            if ode_method == 'euler':
                v = get_velocity(x, t_curr)
                x = x + v * dt
            elif ode_method == 'midpoint':
                k1 = get_velocity(x, t_curr)
                t_mid = t_curr + 0.5 * dt
                k2 = get_velocity(x + 0.5 * dt * k1, t_mid)
                x = x + k2 * dt
            elif ode_method == 'rk4':
                k1 = get_velocity(x, t_curr)
                k2 = get_velocity(x + 0.5 * dt * k1, t_curr + 0.5 * dt)
                k3 = get_velocity(x + 0.5 * dt * k2, t_curr + 0.5 * dt)
                k4 = get_velocity(x + dt * k3, t_curr + dt)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Small clamp for stability
            x = torch.clamp(x, -50, 50)
            
        if return_length: return x * valid_mask.unsqueeze(-1), x0, target_length
        return x * valid_mask.unsqueeze(-1), x0
