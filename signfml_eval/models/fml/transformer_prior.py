"""
Transformer-based Prior: Học drift có cấu trúc thời gian dài hạn
Sử dụng Transformer Encoder thay vì Mamba để đảm bảo ổn định
"""
import torch
import torch.nn as nn


class TransformerPrior(nn.Module):
    """
    Transformer-based Prior cho Flow Matching
    Học prior velocity v_prior(z, t, condition) từ latent sequence
    """
    
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input projection: latent -> hidden
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Time embedding (sinusoidal + MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition projection (text features -> hidden)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: hidden -> velocity (latent_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z, t, condition, mask=None):
        """
        Predict prior velocity v_prior(z, t, condition)
        
        Args:
            z: [B, T, latent_dim] - Latent sequence
            t: [B] - Timesteps
            condition: [B, L, hidden_dim] - Text features
            mask: [B, T] - Valid mask (True = valid, False = padding)
        
        Returns:
            v_prior: [B, T, latent_dim] - Prior velocity
        """
        # Input validation
        if torch.isnan(z).any() or torch.isinf(z).any():
            print(f"⚠️ TransformerPrior: NaN/Inf in z input!")
            z = torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)
        
        if condition is not None and (torch.isnan(condition).any() or torch.isinf(condition).any()):
            print(f"⚠️ TransformerPrior: NaN/Inf in condition input!")
            condition = torch.nan_to_num(condition, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Clamp inputs
        z = torch.clamp(z, -10.0, 10.0)
        if condition is not None:
            condition = torch.clamp(condition, -10.0, 10.0)
        
        B, T, D = z.shape
        
        # 1. Project input and add time embedding
        x = self.input_proj(z)  # [B, T, hidden_dim]
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Clamp and add
        t_emb = torch.clamp(t_emb, -10.0, 10.0)
        x = x + t_emb
        x = torch.clamp(x, -10.0, 10.0)
        
        # 2. Add condition (text features)
        if condition is not None:
            # Global pooling of text features
            cond_global = condition.mean(dim=1).unsqueeze(1)  # [B, 1, hidden_dim]
            cond_emb = self.cond_proj(cond_global)
            cond_emb = torch.clamp(cond_emb, -10.0, 10.0)
            x = x + cond_emb
            x = torch.clamp(x, -10.0, 10.0)
        
        # 3. Prepare mask for Transformer (True = ignore)
        if mask is not None:
            # Our convention: True = valid, False = padding
            # Transformer expects: True = ignore, False = keep
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None
        
        # 4. Pass through Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Clamp after transformer
        x = torch.clamp(x, -100.0, 100.0)
        
        # Check for NaN
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ TransformerPrior: NaN/Inf after transformer!")
            x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # 5. Project to velocity
        v_prior = self.output_proj(x)
        v_prior = torch.clamp(v_prior, -10.0, 10.0)
        
        # Final check
        if torch.isnan(v_prior).any() or torch.isinf(v_prior).any():
            print(f"⚠️ TransformerPrior: NaN/Inf in output! Replacing with zeros.")
            v_prior = torch.nan_to_num(v_prior, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return v_prior


# Alias for backward compatibility
SimpleSSMPrior = TransformerPrior
