"""
Flow Matching Transformer for Text-to-Pose Generation (Stage 2)
Implements Conditional Flow Matching with text conditioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: (B,) timestep in [0, 1]
        Returns:
            (B, dim) embedding
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FlowMatchingTransformer(nn.Module):
    """
    Flow Matching model for Text-to-Latent generation
    
    Architecture:
        - Text Encoder: BERT
        - Flow Model: Transformer with cross-attention to text
        - Output: Velocity prediction v_θ(z_t, t, text)
    """
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        text_encoder: str = 'bert-base-multilingual-cased',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Text encoder (frozen BERT)
        self.text_encoder = BertModel.from_pretrained(text_encoder)
        self.text_dim = self.text_encoder.config.hidden_size  # 768
        
        # Freeze BERT (optional - can fine-tune later)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional encoding (for sequence position)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                text_dim=self.text_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, z_t, t, text_input_ids, text_attention_mask, latent_mask=None):
        """
        Args:
            z_t: (B, T, latent_dim) - noisy latent at time t
            t: (B,) - timestep in [0, 1]
            text_input_ids: (B, L) - BERT input IDs
            text_attention_mask: (B, L) - BERT attention mask
            latent_mask: (B, T) - valid frames mask
        
        Returns:
            v: (B, T, latent_dim) - predicted velocity
        """
        B, T, D = z_t.shape
        
        # 1. Encode text
        with torch.no_grad():
            text_output = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_emb = text_output.last_hidden_state  # (B, L, 768)
        
        # 2. Time embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(B, T, -1)  # (B, T, hidden_dim)
        
        # 3. Project latent
        x = self.latent_proj(z_t)  # (B, T, hidden_dim)
        
        # 4. Add positional encoding
        x = x + self.pos_encoding[:, :T, :]
        
        # 5. Add time embedding
        x = x + t_emb
        
        # 6. Transformer layers
        for layer in self.layers:
            x = layer(x, text_emb, text_attention_mask, latent_mask)
        
        # 7. Final norm
        x = self.final_norm(x)
        
        # 8. Project to velocity
        v = self.out_proj(x)  # (B, T, latent_dim)
        
        return v


class TransformerBlock(nn.Module):
    """
    Transformer block with:
    - Self-attention (for temporal modeling)
    - Cross-attention (to text)
    - Feed-forward network
    """
    def __init__(self, hidden_dim, num_heads, text_dim, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention to text
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True, kdim=text_dim, vdim=text_dim
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, text_emb, text_mask, latent_mask=None):
        """
        Args:
            x: (B, T, hidden_dim)
            text_emb: (B, L, text_dim)
            text_mask: (B, L)
            latent_mask: (B, T)
        """
        # Self-attention
        attn_mask = None
        if latent_mask is not None:
            # Create attention mask (True = ignore)
            attn_mask = ~latent_mask.unsqueeze(1).expand(-1, latent_mask.shape[1], -1)
        
        x_attn, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + x_attn)
        
        # Cross-attention to text
        text_attn_mask = ~text_mask.unsqueeze(1).expand(-1, x.shape[1], -1)
        x_cross, _ = self.cross_attn(x, text_emb, text_emb, attn_mask=text_attn_mask)
        x = self.norm2(x + x_cross)
        
        # FFN
        x_ffn = self.ffn(x)
        x = self.norm3(x + x_ffn)
        
        return x


if __name__ == "__main__":
    # Test
    model = FlowMatchingTransformer(
        latent_dim=256,
        hidden_dim=512,
        num_layers=6,
        num_heads=8
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Test forward
    B, T, L = 2, 100, 128
    z_t = torch.randn(B, T, 256)
    t = torch.rand(B)
    text_ids = torch.randint(0, 30000, (B, L))
    text_mask = torch.ones(B, L, dtype=torch.bool)
    latent_mask = torch.ones(B, T, dtype=torch.bool)
    
    v = model(z_t, t, text_ids, text_mask, latent_mask)
    
    print(f"Input: {z_t.shape}")
    print(f"Output: {v.shape}")
    print("✅ Model test passed!")