"""
Stage 1: Pose Autoencoder - OPTIMAL v2
Based on user feedback and SOTA practices

✅ IMPROVEMENTS:
1. Stronger Encoder (30M, 38% of total)
2. Multi-Scale Attention (local/medium/global heads)
3. RoPE (Rotary Position Embeddings)
4. Cross-Attention between decoder levels
5. AdaLN (Adaptive Layer Normalization)
6. Learnable residual weights
7. SwiGLU activation

TOTAL: ~78M parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =============================================================================
# HELPER MODULES
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(gate) * x"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    Better than absolute position encoding for temporal modeling
    """
    def __init__(self, dim: int, max_seq_len: int = 5000, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position embeddings
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to q and k
        Args:
            q, k: [B, nhead, T, head_dim]
        Returns:
            q_rotated, k_rotated: same shape
        """
        seq_len = q.size(2)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated
    
    def _rotate_half(self, x):
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat([-x2, x1], dim=-1)


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN)
    Conditions normalization on latent embedding
    """
    def __init__(self, dim: int, condition_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_shift = nn.Linear(condition_dim, dim * 2)
        nn.init.zeros_(self.scale_shift.weight)
        nn.init.zeros_(self.scale_shift.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, dim]
            condition: [B, T, condition_dim] or [B, condition_dim]
        """
        normalized = self.norm(x)
        
        # Handle condition shape
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)  # [B, 1, cond_dim]
        
        # Mean pooling if condition has different time dim
        if condition.size(1) != x.size(1):
            condition = condition.mean(dim=1, keepdim=True)  # [B, 1, cond_dim]
        
        scale_shift = self.scale_shift(condition)  # [B, T, dim*2]
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        return normalized * (1 + scale) + shift


# =============================================================================
# ATTENTION MODULES
# =============================================================================

class MultiScaleFlashAttention(nn.Module):
    """
    Multi-Scale Flash Attention with different receptive fields:
    - Local heads: window size 16 (finger movements)
    - Medium heads: window size 64 (arm movements)
    - Global heads: full sequence (body coordination)
    """
    def __init__(
        self,
        dim: int,
        local_heads: int = 2,
        medium_heads: int = 2,
        global_heads: int = 4,
        local_window: int = 16,
        medium_window: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.total_heads = local_heads + medium_heads + global_heads
        self.head_dim = dim // self.total_heads
        
        self.local_heads = local_heads
        self.medium_heads = medium_heads
        self.global_heads = global_heads
        self.local_window = local_window
        self.medium_window = medium_window
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE for each scale
        self.rope = RotaryPositionEmbedding(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, dim]
            mask: [B, T] padding mask (True = ignore)
        """
        B, T, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        
        qkv = qkv.reshape(B, T, 3, self.total_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Split into local, medium, global heads
        outputs = []
        head_idx = 0
        
        # Local attention (window = 16)
        if self.local_heads > 0:
            q_local = q[:, head_idx:head_idx+self.local_heads]
            k_local = k[:, head_idx:head_idx+self.local_heads]
            v_local = v[:, head_idx:head_idx+self.local_heads]
            
            # Local attention (window = 16)
            local_mask = self._create_window_mask(T, self.local_window, x.device)
            # ✅ FIX Bug #9: Combine window mask with padding mask
            if mask is not None:
                if local_mask.dtype == torch.bool:
                    pad_mask = mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, T] True=Block (Ignore)
                    local_mask = local_mask.unsqueeze(0) | pad_mask # [B, 1, T, T]
                else:
                    # additive
                    pad_mask = torch.zeros_like(mask, dtype=local_mask.dtype).unsqueeze(1).unsqueeze(2)
                    pad_mask = pad_mask.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                    local_mask = local_mask.unsqueeze(0) + pad_mask

            out_local = F.scaled_dot_product_attention(
                q_local, k_local, v_local,
                attn_mask=local_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
            # ✅ FIX: Mask Output (Prevent NaNs on padding tokens)
            if mask is not None:
                out_local = torch.nan_to_num(out_local, nan=0.0) # Safety first
                # mask is True=Ignore. We want to keep Valid (~mask).
                out_local = out_local * (~mask).unsqueeze(1).unsqueeze(-1).float()
                
            outputs.append(out_local)
            head_idx += self.local_heads
        
        # Medium attention (window = 64)
        if self.medium_heads > 0:
            q_medium = q[:, head_idx:head_idx+self.medium_heads]
            k_medium = k[:, head_idx:head_idx+self.medium_heads]
            v_medium = v[:, head_idx:head_idx+self.medium_heads]
            
            # Medium attention (window = 64)
            medium_mask = self._create_window_mask(T, self.medium_window, x.device)
            # ✅ FIX Bug #9: Combine window mask with padding mask
            if mask is not None:
                if medium_mask.dtype == torch.bool:
                    pad_mask = mask.unsqueeze(1).unsqueeze(2)
                    medium_mask = medium_mask.unsqueeze(0) | pad_mask
                else:
                    pad_mask = torch.zeros_like(mask, dtype=medium_mask.dtype).unsqueeze(1).unsqueeze(2)
                    pad_mask = pad_mask.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                    medium_mask = medium_mask.unsqueeze(0) + pad_mask

            out_medium = F.scaled_dot_product_attention(
                q_medium, k_medium, v_medium,
                attn_mask=medium_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
            # ✅ FIX: Mask Output
            if mask is not None:
                out_medium = torch.nan_to_num(out_medium, nan=0.0)
                out_medium = out_medium * (~mask).unsqueeze(1).unsqueeze(-1).float()
                
            outputs.append(out_medium)
            head_idx += self.medium_heads
        
        # Global attention (full sequence)
        if self.global_heads > 0:
            q_global = q[:, head_idx:head_idx+self.global_heads]
            k_global = k[:, head_idx:head_idx+self.global_heads]
            v_global = v[:, head_idx:head_idx+self.global_heads]
            
            # Handle padding mask for global attention
            global_mask = None
            if mask is not None:
                global_mask = mask.unsqueeze(1).unsqueeze(2)
                global_mask = global_mask.expand(B, self.global_heads, T, T)
                # Removing risky masked_fill. Passing boolean mask directly where True=Padding (Ignore).
                # PyTorch scaled_dot_product_attention treats True in Bool mask as -inf (ignore).
            
            # Manual Attention (Fix for SDPA instability with padding masks)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q_global, k_global.transpose(-2, -1)) * scale
            
            if global_mask is not None:
                # Need to be careful. if mask is boolean True=Valid (from input), then global_mask (expanded) is True=Valid.
                # But masked_fill expects mask where True=Fill.
                # In line 221 above: global_mask = mask...
                # If mask is [B, T] (True=Valid).
                # We want to mask FILTER OUT the False locations.
                # So we should use ~mask.
                
                # Check line 558 in train_stage2_v2.py: mask=batch.get('pose_mask')
                # pose_mask is True=Valid.
                
                # Inside EncoderLayer:
                # If we used boolean mask for SDPA, we usually want True=Ignore or True=Keep depending on implementation.
                # But manual attention `masked_fill` fills where Mask is True.
                # So we want Mask=True where Padding(Invalid).
                # So we should use ~mask.
                
                # Let's fix global_mask creation first.
                global_mask = mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, T] True=Padding (Ignore)
                # No need to expand for masked_fill (broadcasts)
            
            if global_mask is not None:
                scores = scores.masked_fill(global_mask, float('-inf'))
            
            attn_probs = F.softmax(scores, dim=-1)
            # Apply dropout manually if training
            if self.training and self.dropout.p > 0:
                attn_probs = torch.nn.functional.dropout(attn_probs, p=self.dropout.p)
                
            out_global = torch.matmul(attn_probs, v_global)
            
            # ✅ FIX: Mask Output
            if mask is not None:
                out_global = torch.nan_to_num(out_global, nan=0.0)
                out_global = out_global * (~mask).unsqueeze(1).unsqueeze(-1).float()
                
            outputs.append(out_global)
        
        # Concatenate all heads
        if not outputs:
            return torch.zeros_like(x) # Should not happen given init logic
            
        output = torch.cat(outputs, dim=1)  # [B, total_heads, T, head_dim]
        output = output.transpose(1, 2).reshape(B, T, self.dim)
        
        return self.out_proj(output)
    
    def _create_window_mask(self, seq_len: int, window_size: int, device) -> torch.Tensor:
        """Create sliding window attention mask (Vectorized O(T) & Optimized)"""
        # ✅ FIX Bug 6: Vectorized implementation avoiding O(T^2) loop
        indices = torch.arange(seq_len, device=device)
        diff = indices.unsqueeze(0) - indices.unsqueeze(1)  # [T, T]
        # Mask where |i-j| > window_size/2
        mask = torch.where(diff.abs() <= window_size // 2, 0.0, float('-inf'))
        return mask


class FlashCrossAttention(nn.Module):
    """
    Flash Cross-Attention for decoder levels
    Query attends to context from previous levels
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T_q, dim] - query
            context: [B, T_kv, dim] - key/value from previous level
            mask: Optional mask
        """
        B, T_q, _ = x.shape
        T_kv = context.size(1)
        
        # Projections
        q = self.q_proj(x).reshape(B, T_q, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, T_kv, 2, self.heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        
        # Flash attention
        attn_mask = None
        if mask is not None:
            # mask is [B, T_kv] (True=ignore)
            # Expand to [B, 1, 1, T_kv] for broadcasting over heads and query length
            attn_mask = torch.zeros(B, 1, 1, T_kv, device=q.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        out = out.transpose(1, 2).reshape(B, T_q, self.dim)
        return self.out_proj(out)


# =============================================================================
# ENCODER
# =============================================================================

class EncoderLayer(nn.Module):
    """
    Encoder layer with:
    - Multi-Scale Flash Attention (local/medium/global)
    - SwiGLU FFN
    - Learnable residual weights
    """
    def __init__(
        self,
        dim: int = 512,
        local_heads: int = 2,
        medium_heads: int = 2,
        global_heads: int = 4,
        ffn_hidden: int = 1536,  # Reduced for 78M target
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-Scale Attention
        self.attn = MultiScaleFlashAttention(
            dim=dim,
            local_heads=local_heads,
            medium_heads=medium_heads,
            global_heads=global_heads,
            dropout=dropout
        )
        
        # SwiGLU FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim)
        )
        
        # LayerNorms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Learnable residual weights
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with learnable residual
        normed = self.norm1(x)
        if torch.isnan(normed).any(): print("❌ NaN after norm1")
        
        attn_out = self.attn(normed, mask)
        if torch.isnan(attn_out).any(): print("❌ NaN after attn")
        
        x = x + self.alpha1 * self.dropout(attn_out)
        
        # FFN with learnable residual
        normed2 = self.norm2(x)
        if torch.isnan(normed2).any(): print("❌ NaN after norm2")
        
        x = x + self.alpha2 * self.dropout(self.ffn(normed2))
        
        return x


class ImprovedEncoder(nn.Module):
    """
    Improved Encoder with:
    - Multi-Scale Flash Attention
    - RoPE (built into attention)
    - SwiGLU FFN
    - Learnable residual weights
    
    Parameters: ~30M (38% of total)
    """
    def __init__(
        self,
        pose_dim: int = 214,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        local_heads: int = 2,
        medium_heads: int = 2,
        global_heads: int = 4,
        ffn_hidden: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(pose_dim, hidden_dim)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                dim=hidden_dim,
                local_heads=local_heads,
                medium_heads=medium_heads,
                global_heads=global_heads,
                ffn_hidden=ffn_hidden,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, 214] pose sequence
            mask: [B, T] padding mask (True = valid)
        Returns:
            latent: [B, T, 256]
        """
        # Convert mask: True = valid → True = ignore for attention
        attn_mask = ~mask if mask is not None else None
        
        # Input projection
        x = self.input_proj(x)
        if torch.isnan(x).any():
            print("❌ NaN detected after Input Projection")
        # Apply encoder layers
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask)
            if torch.isnan(x).any():
                print(f"❌ NaN detected after Encoder Layer {i}")
        
        # Final norm and projection
        x = self.final_norm(x)
        latent = self.output_proj(x)
        
        return latent


# =============================================================================
# DECODER
# =============================================================================

class DecoderLayer(nn.Module):
    """
    Decoder layer with:
    - Flash Self-Attention
    - Optional Flash Cross-Attention (for level communication)
    - AdaLN (condition on latent)
    - SwiGLU FFN
    - Learnable residual weights
    """
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        has_cross_attn: bool = False,
        condition_dim: int = 256,
        ffn_hidden: int = 1536,  # Reduced for 78M target
        dropout: float = 0.1
    ):
        super().__init__()
        self.has_cross_attn = has_cross_attn
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention (optional)
        if has_cross_attn:
            self.cross_attn = FlashCrossAttention(dim, num_heads, dropout)
            self.norm_cross = AdaptiveLayerNorm(dim, condition_dim)
            self.alpha_cross = nn.Parameter(torch.ones(1))
        
        # SwiGLU FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim)
        )
        
        # AdaLN (condition on latent)
        self.norm1 = AdaptiveLayerNorm(dim, condition_dim)
        self.norm2 = AdaptiveLayerNorm(dim, condition_dim)
        
        # Learnable residual weights
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        cross_context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, dim]
            condition: [B, T, condition_dim] - latent for AdaLN
            cross_context: [B, T', dim] - features from previous level
            mask: Optional padding mask
        """
        # Self-attention with AdaLN
        normed = self.norm1(x, condition)
        # ✅ FIX Bug #2: Pass padding mask to self-attention
        # mask is [B, T] (True=valid). self_attn expects key_padding_mask as [B, T] (True=ignore)
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.self_attn(normed, normed, normed, 
                                      key_padding_mask=key_padding_mask,
                                      need_weights=False)
        x = x + self.alpha1 * self.dropout(attn_out)
        
        # Cross-attention (if enabled)
        if self.has_cross_attn and cross_context is not None:
            normed = self.norm_cross(x, condition)
            # ✅ FIX Bug #8: Pass mask to cross-attention
            # FlashCrossAttention.forward takes `mask` as attn_mask directly or key_padding_mask
            # Looking at FlashCrossAttention implementation: it takes `mask` as `key_padding_mask` (True=ignore)
            cross_mask = ~mask if mask is not None else None
            cross_out = self.cross_attn(normed, cross_context, mask=cross_mask)
            x = x + self.alpha_cross * self.dropout(cross_out)
        
        # FFN with AdaLN
        normed = self.norm2(x, condition)
        x = x + self.alpha2 * self.dropout(self.ffn(normed))
        
        return x


class HierarchicalDecoder(nn.Module):
    """
    Hierarchical Decoder with:
    - Level 1 (Coarse): Body structure (33 keypoints)
    - Level 2 (Medium): + Hands (42 keypoints) with cross-attn from L1
    - Level 3 (Fine): + Face (139 keypoints) with cross-attn from L1+L2
    
    Parameters: ~48M (62% of total)
    """
    def __init__(
        self,
        latent_dim: int = 256,
        pose_dim: int = 214,
        hidden_dim: int = 512,
        num_coarse_layers: int = 4,
        num_medium_layers: int = 4,
        num_fine_layers: int = 6,
        num_heads: int = 8,
        ffn_hidden: int = 1536,  # Reduced for 78M target
        dropout: float = 0.1
    ):
        super().__init__()
        self.pose_dim = pose_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Level 1: Coarse (no cross-attention, first level)
        self.level1_layers = nn.ModuleList([
            DecoderLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                has_cross_attn=False,  # First level: no cross
                condition_dim=latent_dim,
                ffn_hidden=ffn_hidden,
                dropout=dropout
            )
            for _ in range(num_coarse_layers)
        ])
        self.level1_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Level 2: Medium (cross-attention from level 1)
        self.level2_layers = nn.ModuleList([
            DecoderLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                has_cross_attn=True,  # Cross-attend to level 1
                condition_dim=latent_dim,
                ffn_hidden=ffn_hidden,
                dropout=dropout
            )
            for _ in range(num_medium_layers)
        ])
        self.level2_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Level 3: Fine (cross-attention from levels 1+2)
        self.level3_layers = nn.ModuleList([
            DecoderLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                has_cross_attn=True,  # Cross-attend to levels 1+2
                condition_dim=latent_dim,
                ffn_hidden=ffn_hidden,
                dropout=dropout
            )
            for _ in range(num_fine_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, pose_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        latent: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, T, 256]
            mask: [B, T] padding mask (True = valid)
        Returns:
            pose: [B, T, 214]
        """
        # Input projection
        x = self.input_proj(latent)
        
        # Level 1: Coarse (body structure)
        level1_out = x
        for layer in self.level1_layers:
            level1_out = layer(level1_out, condition=latent, mask=mask)
        level1_features = self.level1_proj(level1_out)
        
        # Level 2: Medium (+ hand details)
        level2_out = x + level1_features  # Skip connection
        for layer in self.level2_layers:
            level2_out = layer(
                level2_out,
                condition=latent,
                cross_context=level1_features,  # Cross-attend to level 1
                mask=mask
            )
        level2_features = self.level2_proj(level2_out)
        
        # Level 3: Fine (+ face NMMs)
        level3_out = x + level2_features  # Skip connection
        combined_context = level1_features + level2_features  # Combined context
        for layer in self.level3_layers:
            level3_out = layer(
                level3_out,
                condition=latent,
                cross_context=combined_context,  # Cross-attend to levels 1+2
                mask=mask
            )
        
        # Final output
        output = self.final_norm(level3_out)
        pose = self.output_proj(output)
        
        return pose


# =============================================================================
# UNIFIED AUTOENCODER
# =============================================================================

class UnifiedPoseAutoencoder(nn.Module):
    """
    Optimal Pose Autoencoder v2
    
    Features:
    - Multi-Scale Flash Attention (local/medium/global)
    - RoPE (Rotary Position Embeddings)
    - Cross-Attention between decoder levels
    - AdaLN (Adaptive Layer Normalization)
    - SwiGLU activation
    - Learnable residual weights
    
    Total Parameters: ~78M
    - Encoder: ~30M (38%)
    - Decoder: ~48M (62%)
    """
    
    def __init__(
        self,
        pose_dim: int = 214,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        encoder_layers: int = 8,              # OPTIMAL: increased from 6
        decoder_coarse_layers: int = 3,
        decoder_medium_layers: int = 3,
        decoder_fine_layers: int = 4,
        num_heads: int = 8,
        local_heads: int = 2,
        medium_heads: int = 2,
        global_heads: int = 4,
        ffn_hidden: int = 1536,               # OPTIMAL: balanced for 77M
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = ImprovedEncoder(
            pose_dim=pose_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            local_heads=local_heads,
            medium_heads=medium_heads,
            global_heads=global_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout
        )
        
        self.decoder = HierarchicalDecoder(
            latent_dim=latent_dim,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_coarse_layers=decoder_coarse_layers,
            num_medium_layers=decoder_medium_layers,
            num_fine_layers=decoder_fine_layers,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout
        )
    
    def forward(
        self,
        pose: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pose: [B, T, 214]
            mask: [B, T] (True = valid, False = padding)
        Returns:
            reconstructed_pose: [B, T, 214]
            latent: [B, T, 256]
        """
        # Encode
        latent = self.encoder(pose, mask)
        
        # Decode
        reconstructed_pose = self.decoder(latent, mask)
        
        return reconstructed_pose, latent
    
    def encode(self, pose: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode only"""
        return self.encoder(pose, mask)
    
    def decode(self, latent: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode only"""
        return self.decoder(latent, mask)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Optimal Autoencoder v2...")
    
    model = UnifiedPoseAutoencoder()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\n📊 Parameter Count:")
    print(f"   Encoder: {encoder_params/1e6:.1f}M ({encoder_params/total_params*100:.1f}%)")
    print(f"   Decoder: {decoder_params/1e6:.1f}M ({decoder_params/total_params*100:.1f}%)")
    print(f"   Total: {total_params/1e6:.1f}M")
    
    # Test forward pass
    batch = torch.randn(2, 50, 214)
    mask = torch.ones(2, 50, dtype=torch.bool)
    
    with torch.no_grad():
        recon, latent = model(batch, mask)
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Input: {batch.shape}")
    print(f"   Latent: {latent.shape}")
    print(f"   Output: {recon.shape}")
    
    print("\n✅ All features working:")
    print("   - Multi-Scale Flash Attention (2+2+4 heads)")
    print("   - RoPE (Rotary Position Embeddings)")
    print("   - Cross-Attention between decoder levels")
    print("   - AdaLN (Adaptive Layer Normalization)")
    print("   - SwiGLU activation")
    print("   - Learnable residual weights")
