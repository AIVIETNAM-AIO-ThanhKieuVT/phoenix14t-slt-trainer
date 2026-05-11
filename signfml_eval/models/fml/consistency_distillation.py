"""
Consistency Distillation: Few-step sampling (1-4 steps)
"""
import torch
import torch.nn as nn


class ConsistencyDistillation(nn.Module):
    """
    Consistency Distillation: Distill ODE nhiều bước → 1-4 bước
    """
    
    def __init__(
        self,
        teacher_model,  # LatentFlowMatcher (teacher)
        student_model,  # LatentFlowMatcher (student, có thể là copy)
        num_steps=4  # Số bước của student
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.num_steps = num_steps
    
    def teacher_ode(self, batch, text_features, num_steps=50):
        """Teacher: ODE nhiều bước"""
        self.teacher_model.eval()
        with torch.no_grad():
            latent = self.teacher_model._inference_forward(
                batch, text_features, num_steps=num_steps
            )
        return latent
    
    def student_ode(self, batch, text_features, num_steps=None):
        """Student: ODE ít bước"""
        if num_steps is None:
            num_steps = self.num_steps
        self.student_model.eval()
        latent = self.student_model._inference_forward(
            batch, text_features, num_steps=num_steps
        )
        return latent
    
    def compute_distillation_loss(self, batch, text_features, mask=None):
        """
        Compute distillation loss: ||z_student - z_teacher||^2
        
        Args:
            batch: dict với text_tokens, attention_mask, etc.
            text_features: [B, L, hidden_dim]
            mask: [B, T] - valid mask
        
        Returns:
            loss: scalar
        """
        # Teacher: nhiều bước
        z_teacher = self.teacher_ode(batch, text_features, num_steps=50)  # [B, T, 256]
        
        # Student: ít bước
        z_student = self.student_ode(batch, text_features, num_steps=self.num_steps)  # [B, T, 256]
        
        # Loss: MSE
        loss = (z_student - z_teacher) ** 2
        
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).float()
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss


class RectifiedFlowDistillation(nn.Module):
    """
    Rectified Flow Distillation: Reflow để có path thẳng hơn
    """
    
    def __init__(self, flow_model, num_steps=4):
        super().__init__()
        self.flow_model = flow_model
        self.num_steps = num_steps
    
    def reflow_step(self, z0, z1, num_interpolations=10):
        """
        Reflow: Tạo path thẳng hơn giữa z0 và z1
        
        Args:
            z0: [B, T, D] - start
            z1: [B, T, D] - end
            num_interpolations: số điểm interpolation
        
        Returns:
            z_path: [B, num_interpolations, T, D]
        """
        alphas = torch.linspace(0, 1, num_interpolations, device=z0.device)
        z_path = []
        for alpha in alphas:
            z_alpha = (1 - alpha) * z0 + alpha * z1
            z_path.append(z_alpha)
        return torch.stack(z_path, dim=1)  # [B, num_interpolations, T, D]
    
    def compute_reflow_loss(self, batch, text_features, mask=None):
        """
        Compute reflow loss: học path thẳng hơn
        """
        B = text_features.shape[0]
        T = batch.get('target_length', 50)
        if isinstance(T, torch.Tensor):
            T = T.max().item()
        
        # Sample start and end
        z0 = torch.randn(B, T, 256, device=text_features.device)
        z1 = torch.randn(B, T, 256, device=text_features.device)
        
        # Reflow path
        z_path = self.reflow_step(z0, z1, num_interpolations=self.num_steps + 1)
        
        # Compute velocity along path
        losses = []
        for i in range(self.num_steps):
            z_t = z_path[:, i]  # [B, T, 256]
            z_next = z_path[:, i + 1]  # [B, T, 256]
            t = torch.full((B,), i / self.num_steps, device=text_features.device)
            
            # Predict velocity
            v_pred = self.flow_model.flow_block(z_t, t, text_features, mask=mask)
            
            # Ground truth velocity
            v_gt = (z_next - z_t) * self.num_steps  # Scale by num_steps
            
            # Loss
            loss = (v_pred - v_gt) ** 2
            if mask is not None:
                loss = loss * mask.unsqueeze(-1).float()
                loss = loss.sum() / mask.sum().clamp(min=1)
            else:
                loss = loss.mean()
            
            losses.append(loss)
        
        return torch.stack(losses).mean()

