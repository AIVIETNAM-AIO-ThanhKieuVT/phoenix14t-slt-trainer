#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Prior V2 - With Cross-Attention to Text
====================================================
Author: SignFML Research
Date: 21 Jan 2026

Improvement over V1:
- Uses cross-attention to text features (per-token conditioning)
- Instead of just global text pooling
- AdaLN-Zero for time conditioning
- LayerScale for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import V2 components
try:
    from .flow_matching_v5_shared import (
        AdaLNZero,
        LayerScale,
        FlashCrossAttention,
        RotaryPositionEmbedding,
        SafeTimeEmbedding,
        SinusoidalPositionalEmbedding
    )
except ImportError:
    # Fallback for direct testing
    from flow_matching_v5_shared import (
        AdaLNZero,
        LayerScale,
        FlashCrossAttention,
        RotaryPositionEmbedding,
        SafeTimeEmbedding,
        SinusoidalPositionalEmbedding
    )


class PriorTransformerLayer(nn.Module):
    """
    Transformer layer for Prior with:
    - Self-attention on latent sequence
    - Cross-attention to text features
    - AdaLN-Zero for time conditioning
    - LayerScale for stability
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
        
        # Cross-attention to text (key improvement in V2)
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
        
        # LayerScale for stability
        self.ls_attn = LayerScale(hidden_dim, layer_scale_init)
        self.ls_cross = LayerScale(hidden_dim, layer_scale_init)
        self.ls_ffn = LayerScale(hidden_dim, layer_scale_init)
    
    def forward(
        self,
        x: torch.Tensor,           # [B, T, D]
        t_emb: torch.Tensor,       # [B, D]
        text_features: torch.Tensor,  # [B, L, D]
        rope: Optional[RotaryPositionEmbedding] = None,
        pose_mask: Optional[torch.Tensor] = None,  # [B, T] True=valid
        text_mask: Optional[torch.Tensor] = None   # [B, L] True=valid
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        # ✅ FIX: Ensure masks match current sequence length T
        if pose_mask is not None and pose_mask.shape[1] != T:
            pose_mask = pose_mask[:, :T]
            
        # 1. Self-Attention with AdaLN-Zero
        x_norm, gate_attn = self.adaln_attn(x, t_emb)
        
        # QKV projection
        qkv = self.qkv_proj(x_norm).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        if rope is not None:
            q = rope(q, T)
            k = rope(k, T)
        
        # Prepare mask for attention (True=ignore for SDPA)
        attn_mask = None
        if pose_mask is not None:
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
        
        # Zero-out padding positions
        if pose_mask is not None:
            attn_out = attn_out * pose_mask.unsqueeze(-1).float()
        
        # Residual with gate and LayerScale
        x = x + gate_attn * self.ls_attn(attn_out)
        
        # 2. Cross-Attention to text (KEY V2 FEATURE)
        text_padding_mask = ~text_mask.bool() if text_mask is not None else None
        cross_out = self.cross_attn(
            self.cross_norm(x),
            text_features,
            text_features,
            key_padding_mask=text_padding_mask
        )
        x = x + self.ls_cross(cross_out)
        
        # Zero-out padding positions after cross-attention
        if pose_mask is not None:
            x = x * pose_mask.unsqueeze(-1).float()
        
        # 3. FFN with AdaLN-Zero
        x_norm, gate_ffn = self.adaln_ffn(x, t_emb)
        ffn_out = self.ffn(x_norm)
        x = x + gate_ffn * self.ls_ffn(ffn_out)
        
        # Zero-out padding positions after FFN
        if pose_mask is not None:
            x = x * pose_mask.unsqueeze(-1).float()
        
        return x


class TransformerPriorV2(nn.Module):
    """
    Improved Transformer Prior with Cross-Attention to Text
    
    V2 improvements:
    - Per-token text conditioning via cross-attention (not just global pooling)
    - AdaLN-Zero for time conditioning
    - RoPE for temporal modeling
    - LayerScale for stability
    - Zero-init output for stable start
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        layer_scale_init: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Safe time embedding
        self.time_embed = SafeTimeEmbedding(hidden_dim, max_value=5.0)
        
        # RoPE for temporal modeling
        self.rope = RotaryPositionEmbedding(
            dim=hidden_dim // num_heads,
            max_seq_len=max_seq_len
        )
        
        # Text positional embedding (fixed sinusoidal for cross-attention context)
        self.text_pos_embed = SinusoidalPositionalEmbedding(
            dim=hidden_dim,
            max_seq_len=max_seq_len
        )
        
        # Transformer layers with cross-attention
        self.layers = nn.ModuleList([
            PriorTransformerLayer(
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
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Apply weight init
        self.apply(self._init_weights)

        # Zero-initialize output for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Re-zero AdaLN-Zero modulation layers (overwritten by _init_weights)
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
        z: torch.Tensor,              # [B, T, latent_dim]
        t: torch.Tensor,              # [B]
        text_features: torch.Tensor,  # [B, L, hidden_dim]
        pose_mask: Optional[torch.Tensor] = None,  # [B, T] True=valid
        text_mask: Optional[torch.Tensor] = None   # [B, L] True=valid
    ) -> torch.Tensor:
        """
        Predict prior velocity v_prior(z, t, text)
        
        Returns:
            v_prior: [B, T, latent_dim]
        """
        # Input safety
        z = torch.clamp(z, -50.0, 50.0)
        text_features = torch.clamp(text_features, -50.0, 50.0)
        
        # Check for NaN
        if torch.isnan(z).any():
            z = torch.nan_to_num(z, nan=0.0)
        if torch.isnan(text_features).any():
            text_features = torch.nan_to_num(text_features, nan=0.0)
        
        B, T, _ = z.shape
        
        # ✅ FIX: Ensure masks match current sequence length T
        if pose_mask is not None and pose_mask.shape[1] != T:
            pose_mask = pose_mask[:, :T]
            
        # Project input
        x = self.input_proj(z)  # [B, T, hidden_dim]
        
        # Time embedding
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        
        # Add positional encoding to text features
        text_features = self.text_pos_embed(text_features)
        
        # Process through layers
        for layer in self.layers:
            x = layer(
                x, t_emb, text_features,
                rope=self.rope,
                pose_mask=pose_mask,
                text_mask=text_mask
            )
        
        # Output
        x = self.final_norm(x)
        v_prior = self.output_proj(x)
        
        # Safety clamp
        v_prior = torch.clamp(v_prior, -50.0, 50.0)
        
        if torch.isnan(v_prior).any():
            v_prior = torch.nan_to_num(v_prior, nan=0.0)
            
        return v_prior


# Alias for backward compatibility
TransformerPrior = TransformerPriorV2


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing TransformerPriorV2")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    prior = TransformerPriorV2(
        latent_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in prior.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")
    
    # Test forward pass
    B, T, L = 2, 100, 20
    z = torch.randn(B, T, 256).to(device)
    t = torch.rand(B).to(device)
    text = torch.randn(B, L, 512).to(device)
    pose_mask = torch.ones(B, T).bool().to(device)
    text_mask = torch.ones(B, L).bool().to(device)
    
    v_prior = prior(z, t, text, pose_mask, text_mask)
    
    print(f"Input z: {z.shape}")
    print(f"Input t: {t.shape}")
    print(f"Input text: {text.shape}")
    print(f"Output v_prior: {v_prior.shape}")
    print(f"Output has NaN: {torch.isnan(v_prior).any()}")
    print(f"Output range: [{v_prior.min():.3f}, {v_prior.max():.3f}]")
    
    assert v_prior.shape == z.shape, "Shape mismatch!"
    assert not torch.isnan(v_prior).any(), "NaN in output!"
    
    print("\n✅ TransformerPriorV2 test passed!")
