import math
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union

# =============================================================================
# REPLICATED SLT CORE COMPONENTS (To keep module self-contained)
# =============================================================================

def get_activation(activation_type):
    activations = {
        "relu": nn.ReLU(),
        "softsign": nn.Softsign(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }
    if activation_type not in activations:
        raise ValueError(f"Unknown activation type {activation_type}")
    return activations[activation_type]

class MaskedNorm(nn.Module):
    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError(f"Unsupported Normalization Layer: {norm_type}")
        self.num_features = num_features

    def forward(self, x, mask):
        # During inference (eval mode), it just applies usually
        # But we must preserve the mask logic for correctness if SLT was trained with it
        B, T, C = x.shape
        reshaped = x.reshape([-1, self.num_features])
        normed = self.norm(reshaped)
        return normed.reshape([B, T, C])

class SpatialEmbeddings(nn.Module):
    def __init__(self, embedding_dim, input_size, num_heads, norm_type=None, activation_type=None, scale=False, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.ln = nn.Linear(input_size, embedding_dim)
        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim)
        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)
        self.scale = scale
        if self.scale:
            self.scale_factor = math.sqrt(self.embedding_dim)

    def forward(self, x, mask):
        x = self.ln(x)
        if self.norm_type:
            x = self.norm(x, mask)
        if self.activation_type:
            x = self.activation(x)
        if self.scale:
            x = x * self.scale_factor
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, size: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, size, 2).float() * -(math.log(10000.0) / size))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, emb):
        return emb + self.pe[:, : emb.size(1)]

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, size, dropout=0.1):
        super().__init__()
        assert size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = size // num_heads
        self.k_layer = nn.Linear(size, size)
        self.v_layer = nn.Linear(size, size)
        self.q_layer = nn.Linear(size, size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v, q, mask=None):
        B = k.size(0)
        k = self.k_layer(k).view(B, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_layer(v).view(B, -1, self.num_heads, self.head_size).transpose(1, 2)
        q = self.q_layer(q).view(B, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            # mask: [B, 1, T] -> [B, 1, 1, T]
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_size)
        return self.output_layer(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, ff_size, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, size, ff_size, num_heads, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        o = self.feed_forward(self.dropout(h) + x)
        return o

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, ff_size, num_layers, num_heads, dropout=0.1, emb_dropout=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, ff_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

    def forward(self, x, mask):
        x = self.pe(x)
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)

# =============================================================================
# SLT PERCEPTUAL LOSS WRAPPER
# =============================================================================

class SLTPerceptualLoss(nn.Module):
    """
    SLT Perceptual Loss (Sign Language Transformer Encoder features).
    Provides a semantic bridge by using the hidden representations of a 
    pretrained back-translation model.
    """
    def __init__(self, slt_model_dir: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_dir = Path(slt_model_dir)
        
        # 1. Load Config
        with open(self.model_dir / "config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        
        model_cfg = cfg["model"]
        encoder_cfg = model_cfg["encoder"]
        self.feature_size = cfg["data"]["feature_size"] # 534
        
        # 2. Build Components
        self.sgn_embed = SpatialEmbeddings(
            input_size=self.feature_size,
            **encoder_cfg["embeddings"],
            num_heads=encoder_cfg["num_heads"]
        )
        
        self.encoder = TransformerEncoder(
            **encoder_cfg
        )
        
        # 3. Load Weights
        ckpt_path = self.model_dir / "best.ckpt"
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state"]
        
        # Load sgn_embed
        sgn_embed_dict = {k[10:]: v for k, v in state_dict.items() if k.startswith("sgn_embed.")}
        self.sgn_embed.load_state_dict(sgn_embed_dict)
        
        # Load encoder
        encoder_dict = {k[8:]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_dict)
        
        # 4. Freeze & Eval Mode
        for p in self.parameters():
            p.requires_grad = False
        self.to(device)
        self.eval()
        
        print(f"✅ Loaded SLT Perceptual Encoder from {ckpt_path}")

    def train(self, mode: bool = True):
        """Permanently stay in eval mode to preserve BatchNorm/Dropout behavior."""
        return super().train(False)

    def forward(self, pred_poses_3d: torch.Tensor, gt_poses_3d: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Compute SLT Perceptual Loss (MSE between features).
        
        Args:
            pred_poses_3d: [B, T, 178, 3] or [B, T, 534]
            gt_poses_3d: [B, T, 178, 3] or [B, T, 534]
            mask: [B, T] sequence mask (1 for valid, 0 for padding)
        """
        # 1. Reshape/Flatten if needed
        if pred_poses_3d.dim() == 4:
            pred = pred_poses_3d.flatten(start_dim=2) # [B, T, 534]
            gt = gt_poses_3d.flatten(start_dim=2)
        else:
            pred = pred_poses_3d
            gt = gt_poses_3d
            
        # 2. Extract Features
        # sgn_embed expects x, mask
        # Transformer mask in SLT is usually [B, 1, T] bool
        if mask is not None:
            enc_mask = mask.bool() # [B, T]
        else:
            enc_mask = torch.ones(pred.shape[0], pred.shape[1], device=self.device).bool()

        # sgn_embed.forward(x, mask)
        # Note: SpatialEmbeddings in SLT expects mask [B, 1, T] for MaskedNorm
        norm_mask = enc_mask.unsqueeze(1) # [B, 1, T]
        
        # Forward Pred (Requires Grad through input)
        z_pred = self.sgn_embed(pred, norm_mask)
        f_pred = self.encoder(z_pred, enc_mask)
        
        # Forward GT (No Grad)
        with torch.no_grad():
            z_gt = self.sgn_embed(gt, norm_mask)
            f_gt = self.encoder(z_gt, enc_mask)
            
        # 3. Compute MSE Loss
        diff = (f_pred - f_gt) ** 2
        
        if mask is not None:
            # mask: [B, T] -> [B, T, 1]
            diff = diff * mask.unsqueeze(-1)
            loss = diff.sum() / (mask.sum() * f_pred.size(-1) + 1e-6)
        else:
            loss = diff.mean()
            
        return loss
