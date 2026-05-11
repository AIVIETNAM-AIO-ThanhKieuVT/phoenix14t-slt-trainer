import torch
import torch.nn as nn

class SyncGuidanceHead(nn.Module):
    """
    Compute synchronization score between latent pose and text
    Uses learned projections to measure alignment.
    """
    
    def __init__(self, latent_dim, hidden_dim, dropout=0.1, text_dim=768):
        """
        Args:
            latent_dim: Pose latent dimension
            hidden_dim: Hidden dimension
            text_dim: Text feature dimension (Default 768 for BERT)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # 1. Pose Projection
        self.pose_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Text Projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Score Network
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, latent, text_pooled, mask=None):
        B, T, _ = latent.shape
        
        # Project Pose
        h_pose = self.pose_proj(latent)  # [B, T, H]
        
        # Project Text & Expand Time
        h_text = self.text_proj(text_pooled)  # [B, H]
        h_text = h_text.unsqueeze(1).expand(-1, T, -1)  # [B, T, H]
        
        # Concatenate
        combined = torch.cat([h_pose, h_text], dim=-1)  # [B, T, 2H]
        
        # Compute Score
        scores = self.score_net(combined).squeeze(-1)  # [B, T]
        
        # Apply Mask (Valid=1, Padding=0)
        if mask is not None:
            scores = scores * mask.float()
        
        return scores

class ContrastiveSyncLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, positive_scores, negative_scores):
        loss = torch.relu(self.margin + negative_scores - positive_scores)
        return loss.mean()