#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Matching Components v3 - Bug Fixes
========================================
Author: Kieu Vo
Date: 21 Feb 2026

Changes from V2:
1. ✅ FIX: SafeFlowMatchingLoss uses MSE instead of Huber (standard for Flow Matching)
2. ✅ FIX: Do NOT clamp ground truth velocity (v_gt) in loss — it corrupts training targets
3. ✅ FIX: Removed aggressive input clamping in FlowMatchingBlockV3.forward
4. ✅ FIX: Removed output clamp in FlowMatchingBlockV3 (let loss function handle it)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# 1. Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    From: RoFormer: Enhanced Transformer with Rotary Position Embedding
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for max_seq_len
        self._precompute_cache(max_seq_len)
    
    def _precompute_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply RoPE to input tensor
        
        Args:
            x: [B, H, T, D] - Query or Key tensor
            seq_len: Sequence length (optional, defaults to T)
        
        Returns:
            Rotated tensor [B, H, T, D]
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._precompute_cache(seq_len)
        
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding: x * cos + rotate(x) * sin"""
        # Split in half
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        
        # Rotate
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin


# =============================================================================
# 2. AdaLN-Zero (Adaptive Layer Norm with Zero Initialization)
# =============================================================================

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization
    From: Scalable Diffusion Models with Transformers (DiT)
    
    Predicts scale, shift, and gate from conditioning signal.
    Zero-initialized for stable training start.
    """
    
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # MLP to predict modulation parameters
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 3 * hidden_dim)
        )
        
        # Zero-initialize for identity at start
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] - Input features
            c: [B, D] - Condition (e.g., time embedding)
        
        Returns:
            (normalized_x, gate): Each is a tensor
        """
        # Predict modulation params
        modulation = self.modulation(c)  # [B, 3*D]
        shift, scale, gate = modulation.chunk(3, dim=-1)  # Each [B, D]
        
        # Apply adaptive normalization
        x_norm = self.norm(x)
        x_modulated = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x_modulated, gate.unsqueeze(1)


# =============================================================================
# 3. LayerScale for Stability
# =============================================================================

class LayerScale(nn.Module):
    """
    LayerScale from: Going Deeper with Image Transformers (CaiT)
    Scales residual branch by learnable parameter initialized to small value.
    """
    
    def __init__(self, dim: int, init_value: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# =============================================================================
# 4. Flash Cross-Attention
# =============================================================================

class FlashCrossAttention(nn.Module):
    """
    Cross-Attention using PyTorch's scaled_dot_product_attention
    Supports Flash Attention when available (CUDA)
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_kv, D]
            value: [B, T_kv, D]
            key_padding_mask: [B, T_kv] - True = ignore
        
        Returns:
            [B, T_q, D]
        """
        B, T_q, D = query.shape
        T_kv = key.shape[1]
        
        # Project
        q = self.q_proj(query).view(B, T_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_kv, self.nhead, self.head_dim).transpose(1, 2)
        
        # Prepare attention mask
        attn_mask = None
        if key_padding_mask is not None:
            # [B, T_kv] -> [B, 1, 1, T_kv]: True=ignore → -inf, False=valid → 0.0
            attn_mask = torch.zeros(B, 1, 1, T_kv, device=query.device, dtype=query.dtype)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Scaled dot-product attention (uses Flash Attention when available)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.scale
        )
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        out = self.out_proj(out)
        
        return out


# =============================================================================
# 5. Safe Time Embedding
# =============================================================================

class SafeTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding with bounded output
    Prevents NaN from large embedding values
    """
    
    def __init__(self, hidden_dim: int, max_value: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_value = max_value
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] - Timesteps in [0, 1]
        
        Returns:
            [B, D] - Time embedding
        """
        # Sinusoidal embedding
        half_dim = self.hidden_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)  # [B, D/2]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, D]
        
        # MLP with bounded output
        emb = self.mlp(emb)
        emb = torch.tanh(emb) * self.max_value
        
        return emb


# =============================================================================
# 6. Improved Transformer Layer with AdaLN + LayerScale + RoPE
# =============================================================================

class ImprovedTransformerLayer(nn.Module):
    """
    Transformer layer with SOTA improvements:
    - AdaLN-Zero for time conditioning
    - LayerScale for stability
    - RoPE for position encoding
    - Flash Attention for efficiency
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        layer_scale_init: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # AdaLN for self-attention
        self.adaln_attn = AdaLNZero(hidden_dim, hidden_dim)
        
        # Self-attention projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attn_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        
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
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_features: torch.Tensor,
        rope: Optional[RotaryPositionEmbedding] = None,
        pose_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - Pose features
            t_emb: [B, D] - Combined time/length embedding
            text_features: [B, L, D] - Text features
            rope: RoPE module (optional)
            pose_mask: [B, T] - True = valid
            text_mask: [B, L] - True = valid
            global_cond: [B, D] - Optional supplementary global condition
        
        Returns:
            [B, T, D] - Updated features
        """
        B, T, D = x.shape
        
        # Combine time embedding with global condition if available
        # We can sum them or concatenate and project. Summing is simpler for AdaLN.
        c_emb = t_emb
        if global_cond is not None:
            c_emb = c_emb + global_cond
            
        # 1. Self-Attention with AdaLN
        x_norm, gate_attn = self.adaln_attn(x, c_emb)
        
        # QKV projection
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each [B, T, H, D_h]
        q = q.transpose(1, 2)  # [B, H, T, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        if rope is not None:
            q = rope(q, T)
            k = rope(k, T)
        
        # Prepare mask for attention
        attn_mask = None
        if pose_mask is not None:
            # Our: True=valid, Attn: True=ignore
            attn_mask = (~pose_mask).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.num_heads, T, -1).float()
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))
        
        # Scaled dot-product attention
        dropout_p = self.attn_dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.attn_out_proj(attn_out)
        
        # Zero-out padding positions to prevent noise accumulation
        if pose_mask is not None:
            attn_out = attn_out * pose_mask.unsqueeze(-1).float()
        
        # Residual with gate and LayerScale
        x = x + gate_attn * self.ls_attn(attn_out)
        
        # 2. Cross-Attention to text
        text_padding_mask = ~text_mask if text_mask is not None else None
        cross_out = self.cross_attn(
            self.cross_norm(x), 
            text_features, 
            text_features,
            key_padding_mask=text_padding_mask
        )
        x = x + self.ls_cross(cross_out)
        
        # 3. FFN with AdaLN
        x_norm, gate_ffn = self.adaln_ffn(x, c_emb)
        ffn_out = self.ffn(x_norm)
        x = x + gate_ffn * self.ls_ffn(ffn_out)
        
        return x


# =============================================================================
# 7. Flow Matching Block V3 (Fixed)
# =============================================================================

class FlowMatchingBlockV3(nn.Module):
    """
    Flow Matching Block V3 — Fixes from V2:
    - ✅ Removed aggressive input clamping (was -50/+50)
    - ✅ Removed output clamping (let loss function handle it)
    - Kept RoPE, AdaLN-Zero, LayerScale, Flash Attention, Length Embedding
    """
    
    def __init__(
        self,
        data_dim: int = 256,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        layer_scale_init: float = 0.1
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        # Safe time embedding
        self.time_embed = SafeTimeEmbedding(hidden_dim, max_value=5.0)
        
        # Condition projection (text: condition_dim -> hidden_dim)
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Length Embedding (SignFlow-style) for length-aware generation
        self.max_length = max_seq_len
        self.length_embed = nn.Embedding(max_seq_len + 1, hidden_dim)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(
            dim=hidden_dim // num_heads,
            max_seq_len=max_seq_len
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ImprovedTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim * 4,
                dropout=dropout,
                layer_scale_init=layer_scale_init
            )
            for _ in range(num_layers)
        ])
        
        # Final norm and output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        
        # General init FIRST, then explicit zero-inits AFTER.
        self.apply(self._init_weights)
        
        # Zero-initialize output projection for stable start (DiT design)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Re-zero AdaLN-Zero modulation layers
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
        global_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict velocity v(x, t, condition)
        
        Args:
            x: [B, T, data_dim] - Noisy latent
            t: [B] - Timesteps in [0, 1]
            condition: [B, L, condition_dim] - Text features
            x_mask: [B, T] - True = valid
            condition_mask: [B, L] - True = valid
            global_cond: [B, hidden_dim] - Supplementary global condition
        
        Returns:
            v: [B, T, data_dim] - Predicted velocity
        """
        B, T, _ = x.shape
        
        # Ensure masks match current sequence length T
        if x_mask is not None and x_mask.shape[1] != T:
            x_mask = x_mask[:, :T]
        
        # ✅ FIX V3: Removed aggressive input clamping (-50/+50)
        # Latents are standardized N(0,1), conditions from BERT are bounded.
        # Only use nan_to_num for safety, not clamping.
        x = torch.nan_to_num(x, nan=0.0)
        condition = torch.nan_to_num(condition, nan=0.0)
        
        # Project input and condition
        h = self.input_proj(x)
        cond = self.cond_proj(condition)
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Add length embedding if provided
        if seq_length is not None:
            # Clamp to valid range
            seq_length_clamped = seq_length.clamp(0, self.max_length).long()
            len_emb = self.length_embed(seq_length_clamped)  # [B, hidden_dim]
            t_emb = t_emb + len_emb  # Combine with time embedding
        
        # Process through transformer layers
        for layer in self.layers:
            h = layer(
                h, t_emb, cond,
                rope=self.rope,
                pose_mask=x_mask,
                text_mask=condition_mask,
                global_cond=global_cond
            )
        
        # Output
        h = self.final_norm(h)
        v = self.output_proj(h)
        
        # ✅ FIX V3: Removed output clamping (-50/+50)
        # Let the loss function handle outliers instead of killing gradients here
        
        return v


# =============================================================================
# 8. Flow Matching Scheduler (unchanged)
# =============================================================================

class FlowMatchingScheduler:
    """
    Straight-line interpolation scheduler for Flow Matching
    Uses optimal transport path: z_t = (1-t)*x0 + t*x1
    """
    
    def get_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate between noise (x0) and data (x1)"""
        t = t.view(-1, 1, 1)
        return (1 - t) * x0 + t * x1
    
    def get_velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Target velocity = x1 - x0 (constant for straight line)"""
        return x1 - x0
    
    def add_noise(
        self, 
        x1: torch.Tensor, 
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample noisy state z_t and target velocity
        
        Returns:
            (z_t, v_gt, x0): Noisy state, target velocity, noise
        """
        if noise is None:
            noise = torch.randn_like(x1)
        
        x0 = noise  # x0 = noise
        
        # Interpolation
        z_t = self.get_path(x0, x1, t)
        
        # Target velocity (constant for straight-line OT)
        v_gt = self.get_velocity(x0, x1, t)
        
        return z_t, v_gt, x0


# =============================================================================
# 9. Safe Flow Matching Loss V3 (FIXED)
# =============================================================================

class SafeFlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss V3 — Critical fixes:
    1. ✅ Uses MSE (not Huber) — standard for all Flow Matching papers
    2. ✅ Does NOT clamp v_gt — ground truth must not be corrupted
    3. ✅ Only applies nan_to_num for safety, minimal clamping on v_pred only
    """
    
    def __init__(self, max_loss: float = 1000.0, use_huber: bool = False, huber_delta: float = 10.0):
        super().__init__()
        self.max_loss = max_loss
        # ✅ V3: use_huber parameter is kept for compatibility but defaults to False
        self.use_huber = use_huber
        self.huber_delta = huber_delta
    
    def forward(
        self, 
        v_pred: torch.Tensor, 
        v_gt: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            v_pred: [B, T, D] - Predicted velocity
            v_gt: [B, T, D] - Target velocity
            mask: [B, T] - True = valid
        
        Returns:
            Scalar loss
        """
        # ✅ V3 FIX: Only clamp v_pred (model output), NOT v_gt (ground truth)
        v_pred = torch.clamp(v_pred, -10.0, 10.0)
        # v_gt is NOT clamped — it's the ground truth training target!
        
        # Replace NaN/Inf (safety only, should not happen with proper training)
        v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=10.0, neginf=-10.0)
        v_gt = torch.nan_to_num(v_gt, nan=0.0)
        
        # ✅ V3 FIX: Use MSE by default (standard for Flow Matching)
        if self.use_huber:
            loss = F.huber_loss(v_pred, v_gt, reduction='none', delta=self.huber_delta)
        else:
            loss = (v_pred - v_gt) ** 2
        
        # Mean over feature dimension
        loss_per_frame = loss.mean(dim=-1)  # [B, T]
        
        # Apply mask
        if mask is not None:
            mask_float = mask.float()
            masked_loss = loss_per_frame * mask_float
            n_valid = mask_float.sum().clamp(min=1.0)
            loss = masked_loss.sum() / n_valid
        else:
            loss = loss_per_frame.mean()
        
        # Clamp final loss
        return torch.clamp(loss, max=self.max_loss)
