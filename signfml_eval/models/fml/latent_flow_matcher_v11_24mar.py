#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Flow Matcher V11 (24 Mar) — Multi-Scale + Adaptive ODE + Dynamic CFG
============================================================================
Author: SignFML Research
Date: 24 Mar 2026

Based on V5 (28 Feb). Key improvements:
1. Multi-Scale Flow Matching: U-Net style coarse→fine for sequences
2. Adaptive ODE Solver: Midpoint/RK4 instead of Euler
3. Improved Length Predictor: Duration-per-token regression
4. Dynamic CFG: Timestep-dependent guidance + Rescaled CFG
5. Temporal Attention Bias: Local attention window + relative position bias
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from transformers import MBartModel

try:
    from .flow_matching_v5_shared import (
        RotaryPositionEmbedding,
        SinusoidalPositionalEmbedding,
        AdaLNZero,
        LayerScale,
        FlashCrossAttention,
        SafeTimeEmbedding,
        FlowMatchingScheduler,
        SafeFlowMatchingLoss
    )
    from .transformer_prior_v5_28feb import TransformerPriorV2
except ImportError:
    from flow_matching_v5_shared import (
        RotaryPositionEmbedding,
        SinusoidalPositionalEmbedding,
        AdaLNZero,
        LayerScale,
        FlashCrossAttention,
        SafeTimeEmbedding,
        FlowMatchingScheduler,
        SafeFlowMatchingLoss
    )
    from transformer_prior_v5_28feb import TransformerPriorV2


# =============================================================================
# 1. Temporal-Biased Transformer Layer (Local Attention + Relative Pos Bias)
# =============================================================================

class TemporalBiasedTransformerLayer(nn.Module):
    """
    Transformer layer with temporal inductive bias for sign language sequences.

    Improvements over ImprovedTransformerLayer:
    - Local attention window: nearby frames get higher attention
    - Learned relative position bias: encodes temporal distance
    - RoPE still applied for absolute position awareness
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        layer_scale_init: float = 0.1,
        max_seq_len: int = 512,
        local_window: int = 64,
        use_rel_pos_bias: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.local_window = local_window
        self.use_rel_pos_bias = use_rel_pos_bias

        # AdaLN for self-attention
        self.adaln_attn = AdaLNZero(hidden_dim, hidden_dim)

        # Self-attention projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attn_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Relative position bias (learned, per head)
        if use_rel_pos_bias:
            # Bias table: covers distances from -max_seq_len to +max_seq_len
            self.max_rel_dist = max_seq_len
            self.rel_pos_bias_table = nn.Parameter(
                torch.zeros(num_heads, 2 * max_seq_len + 1)
            )
            nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # Cross-attention (to text)
        self.cross_attn = FlashCrossAttention(hidden_dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # AdaLN for FFN
        self.adaln_ffn = AdaLNZero(hidden_dim, hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # LayerScale
        self.ls_attn = LayerScale(hidden_dim, layer_scale_init)
        self.ls_cross = LayerScale(hidden_dim, layer_scale_init)
        self.ls_ffn = LayerScale(hidden_dim, layer_scale_init)

    def _get_rel_pos_bias(self, T: int, device: torch.device) -> torch.Tensor:
        """Compute relative position bias matrix [num_heads, T, T]."""
        positions = torch.arange(T, device=device)
        # rel_dist[i,j] = j - i, clamped to [-max_rel_dist, max_rel_dist]
        rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # [T, T]
        rel_dist = rel_dist.clamp(-self.max_rel_dist, self.max_rel_dist)
        # Shift to index into table (0 to 2*max_rel_dist)
        rel_dist = rel_dist + self.max_rel_dist
        # Gather bias: [num_heads, T, T]
        bias = self.rel_pos_bias_table[:, rel_dist]  # [H, T, T]
        return bias

    def _get_local_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Create local attention mask: allow only within window. Returns additive mask."""
        positions = torch.arange(T, device=device)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # [T, T]
        # Beyond window gets -inf, within window gets 0
        mask = torch.where(dist <= self.local_window, 0.0, float('-inf'))
        return mask  # [T, T]

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_features: torch.Tensor,
        rope: Optional[RotaryPositionEmbedding] = None,
        pose_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        if pose_mask is not None and pose_mask.shape[1] != T:
            pose_mask = pose_mask[:, :T]

        # 1. Self-Attention with AdaLN + Temporal Bias
        x_norm, gate_attn = self.adaln_attn(x, t_emb)

        qkv = self.qkv_proj(x_norm)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if rope is not None:
            q = rope(q, T)
            k = rope(k, T)

        # Build attention bias: combine padding mask + local window + rel pos bias
        # Start with zeros [B, H, T, T]
        attn_bias = torch.zeros(B, self.num_heads, T, T, device=x.device, dtype=x.dtype)

        # Padding mask
        if pose_mask is not None:
            pad_mask = (~pose_mask).unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, T, -1)
            attn_bias = attn_bias.masked_fill(pad_mask, float('-inf'))

        # Local attention window
        if self.local_window < T:
            local_mask = self._get_local_mask(T, x.device)  # [T, T]
            attn_bias = attn_bias + local_mask.unsqueeze(0).unsqueeze(0)

        # Relative position bias
        if self.use_rel_pos_bias:
            rel_bias = self._get_rel_pos_bias(T, x.device)  # [H, T, T]
            attn_bias = attn_bias + rel_bias.unsqueeze(0)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=dropout_p
        )
        if torch.isnan(attn_out).any():
            attn_out = torch.nan_to_num(attn_out, nan=0.0)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.attn_out_proj(attn_out)

        if pose_mask is not None:
            attn_out = attn_out * pose_mask.unsqueeze(-1).float()

        x = x + gate_attn * self.ls_attn(attn_out)

        # 2. Cross-Attention to text
        text_padding_mask = ~text_mask.bool() if text_mask is not None else None
        cross_out = self.cross_attn(
            self.cross_norm(x), text_features, text_features,
            key_padding_mask=text_padding_mask
        )
        if pose_mask is not None:
            cross_out = cross_out * pose_mask.unsqueeze(-1).float()
        x = x + self.ls_cross(cross_out)

        # 3. FFN with AdaLN
        x_norm, gate_ffn = self.adaln_ffn(x, t_emb)
        ffn_out = self.ffn(x_norm)
        x = x + gate_ffn * self.ls_ffn(ffn_out)

        if pose_mask is not None:
            x = x * pose_mask.unsqueeze(-1).float()

        return x


# =============================================================================
# 2. Multi-Scale Flow Matching Block (U-Net style for sequences)
# =============================================================================

class SequenceDownsample(nn.Module):
    """Downsample temporal sequence by factor 2 using strided conv."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [B, T, D] -> [B, T//2, D]
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            # Downsample mask: [B, T] -> [B, T//2], keep valid if either neighbor valid
            mask_out = mask[:, ::2]
            if mask_out.shape[1] < out.shape[1]:
                mask_out = F.pad(mask_out, (0, out.shape[1] - mask_out.shape[1]), value=False)
            elif mask_out.shape[1] > out.shape[1]:
                mask_out = mask_out[:, :out.shape[1]]
            return out, mask_out
        return out, None


class SequenceUpsample(nn.Module):
    """Upsample temporal sequence by factor 2 using transposed conv."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, target_len: int):
        # x: [B, T, D] -> [B, target_len, D]
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        # Trim or pad to exact target length
        if out.shape[1] > target_len:
            out = out[:, :target_len]
        elif out.shape[1] < target_len:
            out = F.pad(out, (0, 0, 0, target_len - out.shape[1]))
        return out


class MultiScaleFlowBlock(nn.Module):
    """
    U-Net style multi-scale flow matching block for sequences.

    Architecture:
        Fine layers (full resolution) → Downsample → Coarse layers → Upsample → Fine layers
        with skip connections between matching resolutions.
    """

    def __init__(
        self,
        data_dim: int = 256,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        num_fine_layers: int = 3,
        num_coarse_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        layer_scale_init: float = 0.1,
        local_window: int = 64
    ):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # Input/output projections
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.time_embed = SafeTimeEmbedding(hidden_dim, max_value=5.0)
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        self.text_pos_embed = SinusoidalPositionalEmbedding(dim=hidden_dim, max_seq_len=max_seq_len)

        # Length embedding
        self.max_length = max_seq_len
        self.length_embed = nn.Embedding(max_seq_len + 1, hidden_dim)

        # RoPE for fine and coarse levels
        self.rope_fine = RotaryPositionEmbedding(
            dim=hidden_dim // num_heads, max_seq_len=max_seq_len
        )
        self.rope_coarse = RotaryPositionEmbedding(
            dim=hidden_dim // num_heads, max_seq_len=max_seq_len // 2 + 1
        )

        # Encoder (fine → coarse)
        self.encoder_layers = nn.ModuleList([
            TemporalBiasedTransformerLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                ffn_dim=hidden_dim * 4, dropout=dropout,
                layer_scale_init=layer_scale_init,
                max_seq_len=max_seq_len, local_window=local_window
            )
            for _ in range(num_fine_layers)
        ])

        # Downsample
        self.downsample = SequenceDownsample(hidden_dim)

        # Bottleneck (coarse level — global motion)
        self.coarse_layers = nn.ModuleList([
            TemporalBiasedTransformerLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                ffn_dim=hidden_dim * 4, dropout=dropout,
                layer_scale_init=layer_scale_init,
                max_seq_len=max_seq_len // 2 + 1,
                local_window=max_seq_len  # Full attention at coarse level
            )
            for _ in range(num_coarse_layers)
        ])

        # Upsample
        self.upsample = SequenceUpsample(hidden_dim)

        # Skip connection projection (concat skip + upsampled → hidden_dim)
        self.skip_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder (coarse → fine)
        self.decoder_layers = nn.ModuleList([
            TemporalBiasedTransformerLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                ffn_dim=hidden_dim * 4, dropout=dropout,
                layer_scale_init=layer_scale_init,
                max_seq_len=max_seq_len, local_window=local_window
            )
            for _ in range(num_fine_layers)
        ])

        # Final output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, data_dim)

        # Init
        self.apply(self._init_weights)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        for module in self.modules():
            if isinstance(module, AdaLNZero):
                nn.init.zeros_(module.modulation[-1].weight)
                nn.init.zeros_(module.modulation[-1].bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        seq_length: Optional[torch.Tensor] = None,
        mask_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = x.shape

        if x_mask is not None and x_mask.shape[1] != T:
            x_mask = x_mask[:, :T]

        x = torch.clamp(x, -50.0, 50.0)
        condition = torch.clamp(condition, -50.0, 50.0)

        # Project
        h = self.input_proj(x)
        cond = self.cond_proj(condition)
        cond = self.text_pos_embed(cond)
        t_emb = self.time_embed(t)

        if seq_length is not None:
            len_emb = self.length_embed(seq_length.clamp(0, self.max_length).long())
            t_emb = t_emb + len_emb

        if mask_features is not None:
            h = h + mask_features

        # === Encoder (fine level) ===
        skip = None
        for layer in self.encoder_layers:
            h = layer(h, t_emb, cond, rope=self.rope_fine,
                      pose_mask=x_mask, text_mask=condition_mask)
        skip = h  # Save for skip connection

        # === Downsample ===
        h_coarse, coarse_mask = self.downsample(h, x_mask)

        # === Bottleneck (coarse level) ===
        for layer in self.coarse_layers:
            h_coarse = layer(h_coarse, t_emb, cond, rope=self.rope_coarse,
                             pose_mask=coarse_mask, text_mask=condition_mask)

        # === Upsample ===
        h_up = self.upsample(h_coarse, T)

        # === Skip connection ===
        h = self.skip_proj(torch.cat([h_up, skip], dim=-1))

        # === Decoder (fine level) ===
        for layer in self.decoder_layers:
            h = layer(h, t_emb, cond, rope=self.rope_fine,
                      pose_mask=x_mask, text_mask=condition_mask)

        # Output
        h = self.final_norm(h)
        v = self.output_proj(h)
        v = torch.clamp(v, -50.0, 50.0)
        return v


# =============================================================================
# 3. Duration-Aware Length Predictor
# =============================================================================

class DurationAwareLengthPredictor(nn.Module):
    """
    Improved length predictor with per-token duration regression.

    Instead of just predicting total length, predicts duration per text token
    and sums them. This gives better alignment between text and pose length.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.2, init_mean_length: float = 120.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-token duration head: predicts log(frames) for each token
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Global length head (attention pooling → total length)
        self.attn_query = nn.Linear(hidden_dim, 1)
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.init_mean_length = init_mean_length
        self.sigmoid_offset = 4.0
        self._reset_bias()

    def _reset_bias(self):
        target_log = math.log(max(self.init_mean_length, 1.0))
        min_log, max_log = 1.0, 6.5
        ratio = (target_log - min_log) / (max_log - min_log)
        ratio = max(min(ratio, 0.999), 0.001)
        pre_sigmoid = self.sigmoid_offset + math.log(ratio / (1 - ratio))
        with torch.no_grad():
            self.global_head[-1].bias.fill_(pre_sigmoid)

    def forward(self, text_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns pred_log_length [B]."""
        x = self.input_proj(text_features)

        if mask is not None:
            src_key_padding_mask = ~mask
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)

        # Per-token duration (log-space, softplus to ensure positive)
        token_dur = F.softplus(self.duration_head(x).squeeze(-1))  # [B, T]
        if mask is not None:
            token_dur = token_dur * mask.float()

        # Sum durations → total length estimate (in linear space)
        total_dur_from_tokens = token_dur.sum(dim=-1)  # [B]

        # Global prediction (attention pooling)
        attn_scores = self.attn_query(x).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)

        global_pred = self.global_head(pooled).squeeze(-1)
        min_log, max_log = 1.0, 6.5
        global_log_length = min_log + (max_log - min_log) * torch.sigmoid(global_pred - self.sigmoid_offset)

        # Combine: average of global prediction and log(sum_of_token_durations)
        token_log_length = torch.log(total_dur_from_tokens.clamp(min=1.0))
        token_log_length = token_log_length.clamp(min_log, max_log)

        pred_log_length = 0.5 * global_log_length + 0.5 * token_log_length

        return pred_log_length


# =============================================================================
# 4. Main Model: LatentFlowMatcherV11
# =============================================================================

class LatentFlowMatcherV11(nn.Module):
    """
    V11: Multi-Scale Flow + Adaptive ODE + Dynamic CFG + Temporal Bias

    Key improvements over V5:
    1. MultiScaleFlowBlock: U-Net style coarse→fine
    2. Adaptive ODE: Midpoint/RK4 sampling
    3. DurationAwareLengthPredictor: per-token duration
    4. Dynamic CFG: timestep-dependent + rescaled
    5. TemporalBiasedTransformerLayer: local attention + rel pos bias
    """

    def __init__(
        self,
        latent_dim: int = 256,
        text_encoder_name: str = 'facebook/mbart-large-50-many-to-many-mmt',
        hidden_dim: int = 512,
        num_fine_layers: int = 3,
        num_coarse_layers: int = 3,
        num_prior_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_ssm_prior: bool = True,
        cfg_dropout_rate: float = 0.1,
        pred_length_mask_ratio: float = 0.0,
        local_window: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_ssm_prior = use_ssm_prior
        self.cfg_dropout_rate = cfg_dropout_rate
        self.pred_length_mask_ratio = pred_length_mask_ratio

        # mBART-50 encoder (frozen)
        print(f"V11: Loading mBART-50 encoder from: {text_encoder_name}")
        full_mbart = MBartModel.from_pretrained(text_encoder_name)
        self.text_encoder = full_mbart.encoder
        del full_mbart

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        text_dim = self.text_encoder.config.hidden_size
        print(f"  Text encoder dim: {text_dim} (mBART-50)")

        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Duration-aware length predictor
        self.length_predictor = DurationAwareLengthPredictor(
            input_dim=text_dim, hidden_dim=hidden_dim,
            num_layers=2, num_heads=4, dropout=dropout
        )
        self.length_loss_fn = nn.SmoothL1Loss()

        # Flow matching
        self.scheduler = FlowMatchingScheduler()
        self.flow_block = MultiScaleFlowBlock(
            data_dim=latent_dim,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_fine_layers=num_fine_layers,
            num_coarse_layers=num_coarse_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            local_window=local_window
        )

        # Prior
        if use_ssm_prior:
            self.ssm_prior = TransformerPriorV2(
                latent_dim=latent_dim, hidden_dim=hidden_dim,
                num_layers=num_prior_layers, num_heads=num_heads,
                dropout=dropout, max_seq_len=max_seq_len
            )
        else:
            self.ssm_prior = None

        self.flow_loss_fn = SafeFlowMatchingLoss(max_loss=1000.0, use_huber=True, huber_delta=5.0)

    def encode_text(self, text_tokens: torch.Tensor,
                    attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=text_tokens, attention_mask=attention_mask)
            text_encoder_feats = outputs.last_hidden_state
            text_encoder_feats = text_encoder_feats * attention_mask.unsqueeze(-1).float()
        text_feats = self.text_proj(text_encoder_feats)
        return text_feats, attention_mask, text_encoder_feats

    def get_condition(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if 'raw_text_feats' in batch:
            raw_text_feats = batch['raw_text_feats']
            text_mask = batch['attention_mask']
            text_feats = self.text_proj(raw_text_feats)
            return text_feats, text_mask, raw_text_feats
        text_feats, text_mask, raw_text_feats = self.encode_text(batch['text_tokens'], batch['attention_mask'])
        return text_feats, text_mask, raw_text_feats

    def forward(self, batch: Dict[str, torch.Tensor],
                gt_latent: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None,
                raw_text_feats: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute flow matching and length prediction losses."""
        if condition is None:
            condition, text_mask, raw_text_feats = self.get_condition(batch)
        B, T_text, _ = condition.shape
        D = self.latent_dim

        # Length prediction loss
        gt_lengths = batch['seq_lengths'].float()
        gt_log_lengths = torch.log(gt_lengths + 1e-6)

        lp_input = raw_text_feats
        if self.training and self.pred_length_mask_ratio > 0:
            mask_indices = torch.rand(raw_text_feats.shape[:2], device=raw_text_feats.device) < self.pred_length_mask_ratio
            lp_input = raw_text_feats.clone()
            lp_input[mask_indices] = 0.0

        pred_log_lengths = self.length_predictor(lp_input, text_mask.bool())
        loss_length = self.length_loss_fn(pred_log_lengths, gt_log_lengths)

        # Flow matching loss
        x1 = gt_latent
        T_pose = x1.shape[1]
        valid_mask = torch.arange(T_pose, device=x1.device)[None, :] < batch['seq_lengths'][:, None]

        ts = torch.rand(B, device=x1.device)
        xt, v_gt, _ = self.scheduler.add_noise(x1, ts)

        # Prior
        loss_prior = torch.tensor(0.0, device=x1.device)
        v_prior = None
        if self.ssm_prior is not None:
            v_prior = self.ssm_prior(xt, ts, condition, pose_mask=valid_mask, text_mask=text_mask)
            prior_diff = (v_prior - v_gt.detach()) ** 2
            loss_prior = (prior_diff * valid_mask.unsqueeze(-1).float()).sum() / (valid_mask.sum() * D).clamp(min=1.0)

        # Main flow
        v_flow = self.flow_block(xt, ts, condition, x_mask=valid_mask, condition_mask=text_mask)
        v_flow = torch.nan_to_num(v_flow, nan=0.0)

        if v_prior is not None:
            lambda_t = 0.5 * (1.0 - ts).view(-1, 1, 1)
            v_target = v_gt - lambda_t * v_prior.detach()
        else:
            v_target = v_gt
        loss_flow = self.flow_loss_fn(v_flow, v_target, mask=valid_mask)

        # Estimate x1 for auxiliary losses
        v_total = v_flow
        if v_prior is not None:
            lambda_t_inf = 0.5 * (1.0 - ts).view(-1, 1, 1)
            v_total = v_flow + lambda_t_inf * v_prior.detach()
        est_x1 = xt + (1.0 - ts.view(-1, 1, 1)) * v_total

        return {
            'loss_flow': loss_flow,
            'loss_length': loss_length,
            'loss_prior': loss_prior,
            'est_x1': est_x1,
            'timesteps': ts,
            'valid_mask': valid_mask,
        }

    # =========================================================================
    # Sampling with Adaptive ODE + Dynamic CFG
    # =========================================================================

    def _dynamic_cfg_scale(self, t: float, cfg_scale: float) -> float:
        """
        Dynamic CFG: higher guidance early (structure), lower late (details).

        Schedule: cfg(t) = cfg_scale * (1 - t)^0.5
        At t=0: full guidance. At t=1: minimal guidance.
        """
        return cfg_scale * ((1.0 - t) ** 0.5)

    def _rescale_cfg_velocity(self, v_cond: torch.Tensor, v_cfg: torch.Tensor,
                               rescale_phi: float = 0.7) -> torch.Tensor:
        """
        Rescaled CFG (Imagen-style): prevent over-saturation.

        Normalize the CFG output to match the magnitude of the conditional output,
        blended with the raw CFG output.
        """
        std_cond = v_cond.std(dim=-1, keepdim=True).clamp(min=1e-6)
        std_cfg = v_cfg.std(dim=-1, keepdim=True).clamp(min=1e-6)
        v_rescaled = v_cfg * (std_cond / std_cfg)
        return rescale_phi * v_rescaled + (1 - rescale_phi) * v_cfg

    def _velocity_step(self, x: torch.Tensor, t_val: float, condition: torch.Tensor,
                       valid_mask: torch.Tensor, text_mask: torch.Tensor,
                       cfg_scale: float, null_cond: Optional[torch.Tensor],
                       null_text_mask: Optional[torch.Tensor],
                       use_dynamic_cfg: bool = True,
                       rescale_phi: float = 0.7) -> torch.Tensor:
        """Compute velocity at a single timestep with CFG + Prior."""
        B = x.shape[0]
        t = torch.full((B,), t_val, device=x.device)

        v_cond = self.flow_block(x, t, condition, x_mask=valid_mask, condition_mask=text_mask)

        # CFG
        if cfg_scale != 1.0 and null_cond is not None:
            effective_cfg = self._dynamic_cfg_scale(t_val, cfg_scale) if use_dynamic_cfg else cfg_scale
            v_uncond = self.flow_block(x, t, null_cond, x_mask=valid_mask, condition_mask=null_text_mask)
            v = v_uncond + effective_cfg * (v_cond - v_uncond)
            # Rescaled CFG
            if rescale_phi > 0:
                v = self._rescale_cfg_velocity(v_cond, v, rescale_phi)
        else:
            v = v_cond

        # Prior guidance
        if self.ssm_prior is not None:
            v_prior = self.ssm_prior(x, t, condition, pose_mask=valid_mask, text_mask=text_mask)
            lambda_t = 0.5 * (1.0 - t_val)
            v = v + lambda_t * v_prior

        v = torch.nan_to_num(v, nan=0.0)
        v = torch.clamp(v, -10.0, 10.0)
        return v

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
        return_length: bool = False,
        ode_method: str = 'midpoint',
        use_dynamic_cfg: bool = True,
        rescale_phi: float = 0.7
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample with adaptive ODE solver and dynamic CFG.

        Args:
            ode_method: 'euler', 'midpoint', or 'rk4'
            use_dynamic_cfg: If True, cfg_scale varies with timestep
            rescale_phi: Rescaled CFG strength (0=off, 0.7=default)
        """
        if condition is None:
            condition, text_mask, raw_text_feats = self.get_condition(batch)
        B = condition.shape[0]

        # Predict length
        if target_length is None:
            pred_log_len = self.length_predictor(raw_text_feats, text_mask.bool())
            target_length = torch.exp(pred_log_len).round().long()
            target_length = torch.clamp(target_length, min=10, max=512)

        max_len = target_length.max().item()
        valid_mask = torch.arange(max_len, device=condition.device)[None, :] < target_length[:, None]

        # Initial noise
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
        x = x0.clone()

        # Prepare null condition for CFG
        null_cond = None
        null_text_mask = None
        if cfg_scale != 1.0:
            null_cond = torch.zeros_like(condition)
            null_text_mask = torch.zeros(B, condition.shape[1], device=condition.device, dtype=text_mask.dtype)
            null_text_mask[:, 0] = True

        # ODE integration
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i * dt

            if ode_method == 'euler':
                v = self._velocity_step(x, t_val, condition, valid_mask, text_mask,
                                         cfg_scale, null_cond, null_text_mask,
                                         use_dynamic_cfg, rescale_phi)
                x = x + v * dt

            elif ode_method == 'midpoint':
                # k1
                v1 = self._velocity_step(x, t_val, condition, valid_mask, text_mask,
                                          cfg_scale, null_cond, null_text_mask,
                                          use_dynamic_cfg, rescale_phi)
                x_mid = x + v1 * (dt / 2)
                x_mid = torch.clamp(x_mid, -20.0, 20.0)
                # k2
                v2 = self._velocity_step(x_mid, t_val + dt / 2, condition, valid_mask, text_mask,
                                          cfg_scale, null_cond, null_text_mask,
                                          use_dynamic_cfg, rescale_phi)
                x = x + v2 * dt

            elif ode_method == 'rk4':
                # Runge-Kutta 4th order
                k1 = self._velocity_step(x, t_val, condition, valid_mask, text_mask,
                                          cfg_scale, null_cond, null_text_mask,
                                          use_dynamic_cfg, rescale_phi)
                x2 = torch.clamp(x + k1 * (dt / 2), -20.0, 20.0)
                k2 = self._velocity_step(x2, t_val + dt / 2, condition, valid_mask, text_mask,
                                          cfg_scale, null_cond, null_text_mask,
                                          use_dynamic_cfg, rescale_phi)
                x3 = torch.clamp(x + k2 * (dt / 2), -20.0, 20.0)
                k3 = self._velocity_step(x3, t_val + dt / 2, condition, valid_mask, text_mask,
                                          cfg_scale, null_cond, null_text_mask,
                                          use_dynamic_cfg, rescale_phi)
                x4 = torch.clamp(x + k3 * dt, -20.0, 20.0)
                k4 = self._velocity_step(x4, t_val + dt, condition, valid_mask, text_mask,
                                          cfg_scale, null_cond, null_text_mask,
                                          use_dynamic_cfg, rescale_phi)
                x = x + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

            x = torch.nan_to_num(x, nan=0.0)
            x = torch.clamp(x, -20.0, 20.0)

            if torch.isnan(x).any():
                print(f"  NaN at step {i}!")
                break

        if return_length:
            return x * valid_mask.unsqueeze(-1), x0, target_length
        return x * valid_mask.unsqueeze(-1), x0

    def get_log_prob(self, latents: torch.Tensor, batch: Dict[str, torch.Tensor],
                     x0: Optional[torch.Tensor] = None,
                     condition: Optional[torch.Tensor] = None,
                     K: int = 4,
                     pred_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Heuristic log-prob for SCST."""
        if condition is None:
            condition, text_mask, _ = self.get_condition(batch)
        else:
            text_mask = batch['attention_mask'].bool()

        B, T_pose, D = latents.shape
        device = latents.device
        seq_lens = pred_lengths if pred_lengths is not None else batch['seq_lengths']
        valid_mask = torch.arange(T_pose, device=device)[None, :] < seq_lens[:, None]

        if x0 is None:
            x0 = torch.randn_like(latents)

        all_mse = []
        for k in range(K):
            t_min, t_max = k / K, (k + 1) / K
            ts = t_min + (t_max - t_min) * torch.rand(B, device=device)

            xt = (1 - ts[:, None, None]) * x0 + ts[:, None, None] * latents
            vt_gt = latents - x0
            vt_pred = self.flow_block(xt, ts, condition, x_mask=valid_mask, condition_mask=text_mask)
            vt_pred = torch.clamp(vt_pred, -10.0, 10.0)

            mse = ((vt_pred - vt_gt) ** 2) * valid_mask.unsqueeze(-1).float()
            mse = mse.sum(dim=(1, 2)) / (valid_mask.sum(dim=1) * D).clamp(min=1.0)
            all_mse.append(mse)

        avg_mse = torch.stack(all_mse, dim=0).mean(dim=0)
        return -avg_mse
