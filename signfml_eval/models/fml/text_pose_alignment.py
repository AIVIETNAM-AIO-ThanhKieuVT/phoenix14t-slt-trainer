#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-Pose Alignment Loss (CLIP-Style)
======================================
Contrastive loss to align text embeddings with pose latents.

Based on CLIP (Radford et al., 2021) and adapted for sign language production.

Key idea:
- Text describing a sign should be SIMILAR to the pose latent of that sign
- Text should be DIFFERENT from poses of other signs in the batch

This forces the model to learn text-specific motion patterns rather than
converging to "average" poses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Learnable pooling using attention mechanism.
    Helps the model focus on important tokens (like verbs/glosses).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            mask: [B, L] - True = valid
        """
        # Compute attention scores
        scores = self.attn(x).squeeze(-1)  # [B, L]
        # Mask out padding positions
        scores = scores.masked_fill(~mask, float('-inf'))
        # Softmax to get weights
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        # Weighted sum
        return (x * weights).sum(dim=1)  # [B, D]


class TextPoseAlignmentLoss(nn.Module):
    """
    CLIP-style contrastive loss between text and pose embeddings.
    
    Uses InfoNCE objective to maximize agreement between matching
    text-pose pairs while minimizing agreement with non-matching pairs.
    
    Args:
        text_dim: Dimension of text features (e.g., 512 from hidden_dim)
        pose_dim: Dimension of pose latents (e.g., 256 from latent_dim)
        embed_dim: Shared embedding dimension for alignment
        temperature: Softmax temperature for contrastive loss (lower = harder)
        learnable_temp: Whether temperature is learnable
    """
    
    def __init__(
        self, 
        text_dim: int = 512,   # hidden_dim from text encoder
        pose_dim: int = 256,   # latent_dim from autoencoder
        embed_dim: int = 256,  # shared space dimension
        temperature: float = 0.07,
        learnable_temp: bool = True,
        dropout: float = 0.1   # 🆕 Added dropout for regularization
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        
        # R6: Much heavier dropout to prevent memorization of text-pose pairs
        # With 7060 train samples and 329K params, heavy regularization is critical
        proj_dropout = max(dropout, 0.3)  # R6: minimum 0.3 dropout in projections
        
        # Projection heads (following CLIP design)
        # text_dim → embed_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),   # R6: Heavy dropout
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Attention pooling for text
        self.text_pooling = AttentionPooling(text_dim)
        
        # pose_dim → embed_dim
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),   # R6: Heavy dropout
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Temperature parameter
        if learnable_temp:
            # Initialize with log(1/temperature) for numerical stability
            self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        else:
            self.register_buffer('log_temp', torch.log(torch.tensor(1.0 / temperature)))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for module in [self.text_proj, self.pose_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    @property
    def temperature(self) -> float:
        """Get current temperature value."""
        return torch.exp(-self.log_temp).item()
    
    def forward(
        self, 
        text_features: torch.Tensor,
        pose_latent: torch.Tensor,
        text_mask: torch.Tensor,
        pose_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive alignment loss.
        
        Args:
            text_features: [B, L, D] - Encoded text features
            pose_latent: [B, T, D] - Pose latent representations
            text_mask: [B, L] - True = valid text token
            pose_mask: [B, T] - True = valid pose frame
            
        Returns:
            Scalar loss value
        """
        B = text_features.shape[0]
        
        # Pool text features using attention: [B, L, D] → [B, D]
        text_pooled = self.text_pooling(text_features, text_mask)
        
        # Pool pose latents: [B, T, D] → [B, D]
        pose_valid = pose_mask.unsqueeze(-1).float()
        pose_pooled = (pose_latent * pose_valid).sum(dim=1)
        pose_pooled = pose_pooled / pose_valid.sum(dim=1).clamp(min=1.0)
        
        # Project to shared embedding space
        text_emb = self.text_proj(text_pooled)
        pose_emb = self.pose_proj(pose_pooled)
        
        # L2 normalize embeddings (critical for contrastive learning)
        text_emb = F.normalize(text_emb, dim=-1)
        pose_emb = F.normalize(pose_emb, dim=-1)
        
        # Compute similarity matrix [B, B]
        # Each row i: similarity of text_i with all poses
        # Each col j: similarity of all texts with pose_j
        # ✅ FIX: CLIP uses logit_scale = exp(log_temp), NOT exp(-log_temp)
        # exp(log_temp) ≈ 14.3 → amplifies cosine sim for sharp softmax
        # Previously exp(-log_temp) ≈ 0.07 → killed all contrast → loss stuck at ln(N)
        logit_scale = torch.exp(self.log_temp).clamp(max=100.0)
        logits = torch.matmul(text_emb, pose_emb.T) * logit_scale
        
        # Ground truth: diagonal elements are positive pairs
        labels = torch.arange(B, device=logits.device)
        
        # R6: Label smoothing to prevent memorization of exact text-pose pairs
        # With small dataset (7060 train), hard labels cause rapid overfitting
        label_smoothing = 0.1
        
        # Symmetric InfoNCE loss with label smoothing
        # Text → Pose: P(pose | text)
        loss_t2p = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        
        # Pose → Text: P(text | pose)
        loss_p2t = F.cross_entropy(logits.T, labels, label_smoothing=label_smoothing)
        
        # Average both directions
        loss = (loss_t2p + loss_p2t) / 2.0
        
        return loss


class TextPoseAlignmentLossV2(TextPoseAlignmentLoss):
    """
    Enhanced version with hard negative mining.
    
    Mines semi-hard negatives within the batch for more effective learning.
    """
    
    def __init__(
        self, 
        text_dim: int = 512,
        pose_dim: int = 256,
        embed_dim: int = 256, 
        temperature: float = 0.07,
        learnable_temp: bool = True,
        dropout: float = 0.1,
        margin: float = 0.2,
        hard_neg_weight: float = 0.5
    ):
        super().__init__(text_dim, pose_dim, embed_dim, temperature, learnable_temp, dropout)
        self.margin = margin
        self.hard_neg_weight = hard_neg_weight
    
    def forward(
        self, 
        text_features: torch.Tensor,
        pose_latent: torch.Tensor,
        text_mask: torch.Tensor,
        pose_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward with hard negative mining."""
        B = text_features.shape[0]
        
        # Pool and project (same as parent)
        text_valid = text_mask.unsqueeze(-1).float()
        text_pooled = (text_features * text_valid).sum(dim=1)
        text_pooled = text_pooled / text_valid.sum(dim=1).clamp(min=1.0)
        
        pose_valid = pose_mask.unsqueeze(-1).float()
        pose_pooled = (pose_latent * pose_valid).sum(dim=1)
        pose_pooled = pose_pooled / pose_valid.sum(dim=1).clamp(min=1.0)
        
        text_emb = F.normalize(self.text_proj(text_pooled), dim=-1)
        pose_emb = F.normalize(self.pose_proj(pose_pooled), dim=-1)
        
        # Similarity matrix
        # ✅ FIX: Use exp(log_temp) not exp(-log_temp) — same fix as parent
        logit_scale = torch.exp(self.log_temp).clamp(max=100.0)
        sim = torch.matmul(text_emb, pose_emb.T) * logit_scale
        
        # Standard InfoNCE
        labels = torch.arange(B, device=sim.device)
        loss_nce = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        
        # Hard negative mining: triplet-style loss
        # For each positive pair, find hardest negative
        pos_sim = sim.diag()  # [B] - positive pairs
        
        # Mask out positives
        mask = torch.eye(B, device=sim.device).bool()
        neg_sim = sim.masked_fill(mask, float('-inf'))
        
        # Hardest negative per text
        hard_neg_t2p = neg_sim.max(dim=1)[0]  # [B]
        # Hardest negative per pose
        hard_neg_p2t = neg_sim.max(dim=0)[0]  # [B]
        
        # Triplet margin loss
        loss_triplet_t2p = F.relu(hard_neg_t2p - pos_sim + self.margin).mean()
        loss_triplet_p2t = F.relu(hard_neg_p2t - pos_sim + self.margin).mean()
        loss_triplet = (loss_triplet_t2p + loss_triplet_p2t) / 2
        
        # Combined loss: Apply hard negatives only if batch is large enough
        if B >= 16:
            loss = loss_nce + self.hard_neg_weight * loss_triplet
        else:
            loss = loss_nce
        
        return loss


# Quick test
if __name__ == "__main__":
    print("🧪 Testing TextPoseAlignmentLoss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data (realistic dimensions)
    B, L, T = 8, 20, 100
    text_dim, pose_dim = 512, 256  # Different dimensions!
    
    text_features = torch.randn(B, L, text_dim).to(device)
    pose_latent = torch.randn(B, T, pose_dim).to(device)
    text_mask = torch.ones(B, L).bool().to(device)
    pose_mask = torch.ones(B, T).bool().to(device)
    
    # Test basic loss
    loss_fn = TextPoseAlignmentLoss(
        text_dim=text_dim, pose_dim=pose_dim, embed_dim=256
    ).to(device)
    loss = loss_fn(text_features, pose_latent, text_mask, pose_mask)
    print(f"   Basic Loss: {loss.item():.4f}")
    print(f"   Temperature: {loss_fn.temperature:.4f}")
    
    # Test V2 with hard negatives
    loss_fn_v2 = TextPoseAlignmentLossV2(
        text_dim=text_dim, pose_dim=pose_dim, embed_dim=256
    ).to(device)
    loss_v2 = loss_fn_v2(text_features, pose_latent, text_mask, pose_mask)
    print(f"   V2 Loss: {loss_v2.item():.4f}")
    
    print("✅ Tests passed!")