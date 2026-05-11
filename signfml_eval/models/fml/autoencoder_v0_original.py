"""
Stage 1: Pose Autoencoder
Encoder: Full Pose [B,T,214] → Latent [B,T,256]
Decoder: Latent [B,T,256] → Full Pose [B,T,214]
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding cho Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PoseEncoder(nn.Module):
    """
    Encoder: Pose [B,T,214] → Latent [B,T,256]
    """
    
    def __init__(
        self,
        pose_dim=214,
        latent_dim=256,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(pose_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to latent
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, 214] pose sequence
            mask: [B, T] boolean mask (True = valid, False = padding)
        
        Returns:
            latent: [B, T, 256]
        """
        # Project input
        x = self.input_proj(x)  # [B, T, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask for transformer (True = attend, False = ignore)
        if mask is not None:
            # Convert to attention mask: True = valid, False = padding
            # Transformer expects: True = ignore, False = attend
            attn_mask = ~mask  # [B, T]
        else:
            attn_mask = None
        
        # Transformer encoder
        # Note: nn.TransformerEncoder expects src_key_padding_mask
        # True positions will be ignored
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Project to latent
        latent = self.output_proj(x)  # [B, T, 256]
        
        return latent


class PoseDecoder(nn.Module):
    """
    Decoder: Latent [B,T,256] → Pose [B,T,214]
    Hierarchical: Coarse → Medium → Fine
    """
    
    def __init__(
        self,
        latent_dim=256,
        pose_dim=214,
        hidden_dim=512,
        num_coarse_layers=4,
        num_medium_layers=4,
        num_fine_layers=6,
        num_heads=8,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pose_dim = pose_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len, dropout)
        
        # Coarse decoder
        coarse_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.coarse_decoder = nn.TransformerDecoder(coarse_layer, num_coarse_layers)
        self.coarse_proj = nn.Linear(hidden_dim, hidden_dim) # Bỏ // 2        
        # Medium decoder
        medium_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.medium_decoder = nn.TransformerDecoder(medium_layer, num_medium_layers)
        self.medium_proj = nn.Linear(hidden_dim, hidden_dim) # Bỏ // 2        
        # Fine decoder
        fine_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.fine_decoder = nn.TransformerDecoder(fine_layer, num_fine_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, pose_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, latent, mask=None):
        """
        Args:
            latent: [B, T, 256] latent sequence
            mask: [B, T] boolean mask (True = valid, False = padding)
        
        Returns:
            pose: [B, T, 214]
        """
        # Project input
        x = self.input_proj(latent)  # [B, T, hidden_dim]
        x = self.pos_encoding(x)
        
        # Create attention mask
        if mask is not None:
            attn_mask = ~mask  # [B, T]
        else:
            attn_mask = None
        
        # Coarse decoding (self-attention)
        x_coarse = self.coarse_decoder(x, x, tgt_key_padding_mask=attn_mask, memory_key_padding_mask=attn_mask)
        x_coarse = self.coarse_proj(x_coarse)
        
        # Medium decoding
        x_medium = self.medium_decoder(x, x_coarse, tgt_key_padding_mask=attn_mask, memory_key_padding_mask=attn_mask)
        x_medium = self.medium_proj(x_medium)
        
        # Fine decoding
        x_fine = self.fine_decoder(x, x_medium, tgt_key_padding_mask=attn_mask, memory_key_padding_mask=attn_mask)
        
        # Output projection
        pose = self.output_proj(x_fine)  # [B, T, 214]
        
        return pose


class UnifiedPoseAutoencoder(nn.Module):
    """
    Complete Autoencoder: Encoder + Decoder
    """
    
    def __init__(
        self,
        pose_dim=214,
        latent_dim=256,
        hidden_dim=512,
        encoder_layers=6,
        decoder_coarse_layers=4,
        decoder_medium_layers=4,
        decoder_fine_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.encoder = PoseEncoder(
            pose_dim=pose_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.decoder = PoseDecoder(
            latent_dim=latent_dim,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_coarse_layers=decoder_coarse_layers,
            num_medium_layers=decoder_medium_layers,
            num_fine_layers=decoder_fine_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(self, pose, mask=None):
        """
        Args:
            pose: [B, T, 214]
            mask: [B, T]
        
        Returns:
            reconstructed_pose: [B, T, 214]
            latent: [B, T, 256]
        """
        # Encode
        latent = self.encoder(pose, mask)
        
        # Decode
        reconstructed_pose = self.decoder(latent, mask)
        
        return reconstructed_pose, latent
    
    def encode(self, pose, mask=None):
        """Chỉ encode"""
        return self.encoder(pose, mask)
    
    def decode(self, latent, mask=None):
        """Chỉ decode"""
        return self.decoder(latent, mask)

