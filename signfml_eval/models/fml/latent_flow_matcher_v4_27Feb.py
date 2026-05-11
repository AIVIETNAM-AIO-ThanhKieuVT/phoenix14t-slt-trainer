#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Flow Matcher V4 (27 Feb) — mBART-50 Text Encoder
========================================================
Author: SignFML Research
Date: 27 Feb 2026

Based on LatentFlowMatcherV2 with ONE key change:
    - Text Encoder: mBERT (768-dim) → mBART-50 Large (1024-dim)

All downstream components (FlowMatchingBlockV2, TransformerPriorV2) remain
unchanged — they operate in hidden_dim=512 space via text_proj.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from transformers import MBartModel

# Import V2 components (same as before)
try:
    from .flow_matching_v2 import (
        FlowMatchingBlockV2,
        FlowMatchingScheduler,
        SafeFlowMatchingLoss
    )
    from .transformer_prior_v2 import TransformerPriorV2
    from .sync_guidance import SyncGuidanceHead
except ImportError:
    from flow_matching_v2 import (
        FlowMatchingBlockV2,
        FlowMatchingScheduler,
        SafeFlowMatchingLoss
    )
    from transformer_prior_v2 import TransformerPriorV2
    SyncGuidanceHead = None


class LengthPredictor(nn.Module):
    """Predicts sequence length from text features with attention pooling.
    
    Output is in LOG-SPACE: pred ≈ log(num_frames).
    Use torch.exp(pred) to get raw frame count at inference.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, init_mean_length: float = 120.0,
                 dropout: float = 0.4):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.attn_dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        self.init_mean_length = init_mean_length
        self._reset_bias()

    def _reset_bias(self):
        import math
        log_mean = math.log(max(self.init_mean_length, 1.0))
        with torch.no_grad():
            self.net[-2].bias.fill_(log_mean)

    def reset_weights(self):
        """Full reset of all weights in the predictor."""
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self._reset_bias()
    
    def forward(self, text_features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        text_features = torch.clamp(text_features, -10.0, 10.0)
        text_features = torch.nan_to_num(text_features, nan=0.0)
        attn_scores = self.attn(text_features).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = (text_features * attn_weights.unsqueeze(-1)).sum(dim=1)
        pooled = self.attn_dropout(pooled)
        pred_length = self.net(pooled)
        pred_length = torch.clamp(pred_length, min=0.0, max=6.25)
        return pred_length.squeeze(-1)


class MaskProjector(nn.Module):
    """Projects binary masks [B, T, 1] to hidden dimension [B, T, D]."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, hidden_dim, bias=False)
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        return self.proj(mask.float())


class LatentFlowMatcherV4(nn.Module):
    """
    V4: Uses mBART-50 Large encoder (1024-dim) instead of mBERT (768-dim).
    All other components are identical to V2.
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
        lambda_prior: float = 0.1,
        gamma_guidance: float = 0.01,
        lambda_anneal: bool = True,
        W_PRIOR: float = 0.1,
        W_SYNC: float = 0.1,
        W_LENGTH: float = 0.1,
        cfg_dropout_rate: float = 0.0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_ssm_prior = use_ssm_prior
        self.use_sync_guidance = use_sync_guidance
        self.lambda_prior = lambda_prior
        self.gamma_guidance = gamma_guidance
        self.lambda_anneal = lambda_anneal
        self.cfg_dropout_rate = cfg_dropout_rate
        self.W_PRIOR = W_PRIOR
        self.W_SYNC = W_SYNC
        self.W_LENGTH = W_LENGTH
        
        self.hand_projector = MaskProjector(hidden_dim)
        self.face_projector = MaskProjector(hidden_dim)
        
        # ===== V4 CHANGE: mBART-50 encoder instead of mBERT =====
        print(f"🆕 V4: Loading mBART-50 encoder from: {text_encoder_name}")
        full_mbart = MBartModel.from_pretrained(text_encoder_name)
        self.text_encoder = full_mbart.encoder  # Only use encoder part
        del full_mbart  # Free decoder memory
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # text_dim will be 1024 for mBART-50 Large (auto-detected)
        text_dim = self.text_encoder.config.hidden_size
        print(f"  📐 Text encoder dim: {text_dim} (mBART-50)")
        # ===== END V4 CHANGE =====
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.length_predictor = LengthPredictor(input_dim=text_dim)
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
        
        if use_sync_guidance and SyncGuidanceHead is not None:
            self.sync_head = SyncGuidanceHead(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim // 2,
                dropout=dropout,
                text_dim=hidden_dim
            )
        else:
            self.sync_head = None
        
        self.flow_loss_fn = SafeFlowMatchingLoss(max_loss=1000.0, use_huber=True, huber_delta=5.0)

    def encode_text(self, text_tokens: torch.Tensor, 
                    attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.text_encoder.eval()
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=text_tokens, attention_mask=attention_mask)
        # mBART encoder output is the same interface as BERT
        raw_features = outputs.last_hidden_state  # (B, L, 1024) for mBART
        # Zero-out padding so fresh features match cached features
        raw_features = raw_features * attention_mask.unsqueeze(-1).float()
        text_features = self.text_proj(raw_features)  # (B, L, 512)
        text_mask = attention_mask.bool()
        return text_features, text_mask, raw_features

    def get_lambda_t(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0: t = t.unsqueeze(0)
        B = t.shape[0]
        if self.lambda_anneal:
            lambda_val = self.lambda_prior * (1 - t.float())
            return lambda_val.view(B, 1, 1)
        return torch.full((B, 1, 1), self.lambda_prior, device=t.device, dtype=torch.float32)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        gt_latent: torch.Tensor,
        pose_gt: Optional[torch.Tensor] = None,
        mode: str = 'train',
        num_inference_steps: int = 50,
        cfg_scale: float = 1.5,
        prior_scale: float = 1.0,
        conditions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        text_features, text_mask, bert_features = self.encode_text(
            batch['text_tokens'], batch['attention_mask']
        )
        if mode == 'train':
            return self._train_forward(
                batch, text_features, text_mask, gt_latent, prior_scale,
                bert_features=bert_features, conditions=conditions
            )
        return self.sample_cfg_zero_star(
            batch, steps=num_inference_steps, cfg_scale=cfg_scale,
            text_features=text_features, text_mask=text_mask,
            bert_features=bert_features, conditions=conditions
        )

    def _train_forward(
        self,
        batch: Dict[str, torch.Tensor],
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        gt_latent: torch.Tensor,
        prior_scale: float = 1.0,
        bert_features: Optional[torch.Tensor] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        pred_length_mask_ratio: float = 0.0
    ) -> Dict[str, Any]:
        device = text_features.device
        B, T, D = gt_latent.shape
        mask_features = None
        if conditions is not None:
            mask_features = torch.zeros((B, T, self.hidden_dim), device=device)
            if conditions.get('hand_mask') is not None:
                mask_features += self.hand_projector(conditions['hand_mask'])
            if conditions.get('face_mask') is not None:
                mask_features += self.face_projector(conditions['face_mask'])
            if self.training and self.cfg_dropout_rate > 0:
                mask_drop = torch.rand(B, device=device) < self.cfg_dropout_rate
                if mask_drop.any():
                    mask_features = mask_features.clone()
                    mask_features[mask_drop] = 0.0

        if self.training and self.cfg_dropout_rate > 0:
            drop_mask = torch.rand(B, device=device) < self.cfg_dropout_rate
            if drop_mask.any():
                text_features = text_features.clone()
                text_features[drop_mask] = 0.0

        pred_length = self.length_predictor(bert_features, text_mask)
        target_len = torch.log(batch['seq_lengths'].float() + 1e-6)
        length_loss = F.smooth_l1_loss(pred_length, target_len, beta=0.1)
        # 🆕 Relative length error: penalize percentage deviation (e.g. 20% off)
        pred_frames = torch.exp(pred_length)
        gt_frames = batch['seq_lengths'].float()
        relative_error = ((pred_frames - gt_frames) / gt_frames.clamp(min=10)).abs().mean()
        length_loss = length_loss + 0.5 * relative_error
        length_mae = (pred_frames - gt_frames).abs().mean()
        # 🆕 Cách 3: Occasionally use predicted length for flow mask (forces model to learn good lengths)
        if self.training and pred_length_mask_ratio > 0 and torch.rand(1).item() < pred_length_mask_ratio:
            pred_len = pred_frames.detach().long().clamp(min=10, max=T)
            valid_mask = torch.arange(T, device=device)[None, :] < pred_len[:, None]
        else:
            valid_mask = torch.arange(T, device=device)[None, :] < batch['seq_lengths'][:, None]
        t = torch.rand(B, device=device)
        latent_t, v_gt, _ = self.scheduler.add_noise(gt_latent, t)
        latent_t = torch.nan_to_num(latent_t, nan=0.0)
        v_gt = torch.nan_to_num(v_gt, nan=0.0)

        v_prior = None
        prior_loss = torch.tensor(0.0, device=device)
        if self.use_ssm_prior and self.ssm_prior is not None:
            v_prior = self.ssm_prior(latent_t, t, text_features, pose_mask=valid_mask, text_mask=text_mask)
            prior_diff = (v_prior - v_gt.detach()) ** 2
            prior_loss = (prior_diff * valid_mask.unsqueeze(-1).float()).sum() / (valid_mask.sum() * D).clamp(min=1.0)

        v_flow = self.flow_block(
            latent_t, t, text_features, x_mask=valid_mask, condition_mask=text_mask,
            seq_length=batch['seq_lengths'], mask_features=mask_features
        )
        v_flow = torch.clamp(v_flow, -10.0, 10.0)
        lambda_t = self.get_lambda_t(t) * prior_scale
        v_target = v_gt - lambda_t * v_prior.detach() if v_prior is not None else v_gt

        sync_loss = torch.tensor(0.0, device=device)
        if self.use_sync_guidance and self.sync_head is not None:
            text_valid = text_mask.unsqueeze(-1).float()
            text_pooled = (text_features * text_valid).sum(1) / text_valid.sum(1).clamp(min=1.0)
            pos = (self.sync_head(latent_t, text_pooled, valid_mask) * valid_mask.float()).sum(1) / valid_mask.sum(1).clamp(min=1.0)
            perm = torch.randperm(B, device=device)
            neg = (self.sync_head(latent_t, text_pooled[perm], valid_mask) * valid_mask.float()).sum(1) / valid_mask.sum(1).clamp(min=1.0)
            sync_loss = torch.nan_to_num(F.relu(0.5 + neg - pos).mean(), nan=0.0)

        flow_loss = self.flow_loss_fn(v_flow, v_target, mask=valid_mask)
        total_loss = flow_loss + self.W_PRIOR * prior_loss + self.W_SYNC * sync_loss + self.W_LENGTH * length_loss
        v_pred_live = v_flow + lambda_t * v_prior if v_prior is not None else v_flow
        v_pred = v_pred_live.detach()

        return {
            'total': total_loss, 'flow': flow_loss, 'prior': prior_loss, 'sync': sync_loss,
            'length': length_loss, 'length_mae': length_mae, 'predicted_latent': latent_t.detach(),
            'velocity_pred': v_pred, 'velocity_target': v_target.detach(),
            'v_flow': v_flow.detach(), 'v_prior': v_prior.detach() if v_prior is not None else None,
            'timestep': t.detach(), 'valid_mask': valid_mask.detach(),
            'v_flow_live': v_flow, 'latent_t_live': latent_t, 'v_pred_live': v_pred_live,
        }

    @torch.no_grad()
    def sample_cfg_zero_star(
        self,
        batch: Dict[str, torch.Tensor],
        steps: int = 50,
        cfg_scale: float = 1.5,
        temperature: float = 1.0,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        max_seq_len: int = 400,
        skip_ratio: float = 0.25,
        cfg_decay: float = 0.3,
        min_cfg_scale: float = 1.0,
        target_length: Optional[torch.Tensor] = None,
        bert_features: Optional[torch.Tensor] = None,
        length_scale: float = 1.0,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        return_length: bool = False
    ) -> torch.Tensor:
        if text_features is None:
            text_features, text_mask, bert_features = self.encode_text(batch['text_tokens'], batch['attention_mask'])
        device = text_features.device
        B = text_features.shape[0]
        if target_length is not None:
            seq_lens = target_length.long().clamp(min=10, max=max_seq_len)
        else:
            if bert_features is None:
                _, _, bert_features = self.encode_text(batch['text_tokens'], batch['attention_mask'])
            log_len_pred = self.length_predictor(bert_features, text_mask)
            seq_lens = (torch.exp(log_len_pred) * length_scale).round().long().clamp(min=10, max=max_seq_len)
        T = int(seq_lens.max().item())
        valid_mask = torch.arange(T, device=device)[None, :] < seq_lens[:, None]
        z = torch.randn(B, T, self.latent_dim, device=device) * temperature
        use_cfg = cfg_scale != 1.0

        mask_features = None
        if conditions is not None:
            mask_features = torch.zeros((B, T, self.hidden_dim), device=device)
            if conditions.get('hand_mask') is not None:
                m = conditions['hand_mask']
                if m.dim() == 2:
                    m = m.unsqueeze(-1)
                if m.shape[1] != T:
                    m = F.pad(m[:, :T], (0, 0, 0, max(0, T - m.shape[1])), value=1.0)
                mask_features += self.hand_projector(m)
            if conditions.get('face_mask') is not None:
                m = conditions['face_mask']
                if m.dim() == 2:
                    m = m.unsqueeze(-1)
                if m.shape[1] != T:
                    m = F.pad(m[:, :T], (0, 0, 0, max(0, T - m.shape[1])), value=1.0)
                mask_features += self.face_projector(m)

        null_text_features = torch.zeros_like(text_features) if use_cfg else None
        null_text_mask = torch.ones_like(text_mask) if use_cfg else None
        skip_steps = int(skip_ratio * steps)
        dt = 1.0 / steps

        for step in range(steps):
            t = torch.full((B,), step / steps, device=device)
            v_cond = self._compute_velocity(z, t, text_features, text_mask, valid_mask, seq_length=seq_lens, mask_features=mask_features)
            if not use_cfg or step < skip_steps:
                v_pred = v_cond
            else:
                v_uncond = self._compute_velocity(z, t, null_text_features, null_text_mask, valid_mask, seq_length=seq_lens, mask_features=mask_features)
                adaptive_scale = max(cfg_scale * (1.0 - cfg_decay * ((step - skip_steps) / max(steps - skip_steps, 1))), min_cfg_scale)
                v_pred = v_uncond + adaptive_scale * (v_cond - v_uncond)
            z = z + v_pred * dt
        out = z * valid_mask.unsqueeze(-1).float()
        if return_length:
            return out, seq_lens
        return out

    def _compute_velocity(
        self, latent: torch.Tensor, t: torch.Tensor, text_features: torch.Tensor, text_mask: torch.Tensor,
        valid_mask: torch.Tensor, seq_length: Optional[torch.Tensor] = None, mask_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        v_flow = self.flow_block(latent, t, text_features, x_mask=valid_mask, condition_mask=text_mask, seq_length=seq_length, mask_features=mask_features)
        v_flow = torch.clamp(v_flow, -10.0, 10.0)
        if self.use_ssm_prior and self.ssm_prior is not None:
            v_flow += self.get_lambda_t(t) * self.ssm_prior(latent, t, text_features, pose_mask=valid_mask, text_mask=text_mask)
        return v_flow

    def sample(self, *args, **kwargs):
        return self.sample_cfg_zero_star(*args, **kwargs)

if __name__ == "__main__":
    print("Testing LatentFlowMatcherV4 (mBART-50)")
