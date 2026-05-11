#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Part Weighted Loss for Sign Language Pose Generation
===========================================================
Author: Kieu Vo - Improved Version
Date: 23 Jan 2026

Addresses the critical issue where hands and face are under-learned
because body keypoints (102 dims) dominate the MSE loss.

Pose structure (214 dims):
- Body: 33 keypoints × 2 = 66 dims (first 66)
- Left Hand: 21 keypoints × 2 = 42 dims (66:108)
- Right Hand: 21 keypoints × 2 = 42 dims (108:150)
- Face: 70 keypoints × 2 = 140 dims (150:290, but we have 214 total)
  Actually: Face uses remaining 64 dims (150:214)

Note: The actual face dim is 64, not 140. Adjusting accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from .structural_losses import bone_length_loss, bone_orientation_loss


class PosePartWeightedLoss(nn.Module):
    """
    Weighted MSE Loss for different body parts in sign language.
    
    Pose dimensions (534 total, SLRTP 3D):
    - Body: [0:24] (8 pts * 3 = 24 dims)
    - Right Hand: [24:87] (21 pts * 3 = 63 dims)
    - Left Hand: [87:150] (21 pts * 3 = 63 dims)
    - Face: [150:534] (128 pts * 3 = 384 dims)
    
    Default weights prioritize hands (55%) over face (30%) and body (15%).
    """
    
    def __init__(
        self,
        pose_dim: int = 534,
        body_dim: int = 24,
        hand_dim: int = 63,
        face_dim: int = 384,
        weight_body: float = 0.15,
        weight_hands: float = 0.55,
        weight_face: float = 0.30,
        use_huber: bool = False,
        huber_delta: float = 1.0,
        use_adaptive_weights: bool = False
    ):
        super().__init__()
        
        self.pose_dim = pose_dim
        self.body_dim = body_dim
        self.hand_dim = hand_dim
        self.face_dim = face_dim
        
        # Validate dimensions
        assert body_dim + 2 * hand_dim + face_dim == pose_dim, \
            f"Dimension mismatch: {body_dim} + 2*{hand_dim} + {face_dim} != {pose_dim}"
        
        # Loss weights
        self.weight_body = weight_body
        self.weight_hands = weight_hands
        self.weight_face = weight_face
        
        # Normalize weights to sum to 1.0 for interpretability
        total = weight_body + weight_hands + weight_face
        self.weight_body /= total
        self.weight_hands /= total
        self.weight_face /= total
        
        # Loss type
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        
        # Adaptive weighting based on motion magnitude
        self.use_adaptive_weights = use_adaptive_weights
        
    def _compute_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute base loss (MSE or Huber)."""
        if self.use_huber:
            return F.huber_loss(pred, target, reduction='none', delta=self.huber_delta)
        else:
            return (pred - target) ** 2
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_parts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute weighted loss across body parts.
        
        Args:
            pred: [B, T, 214] - Predicted poses
            target: [B, T, 214] - Target poses
            mask: [B, T] - True = valid, False = padding
            return_parts: If True, return (total_loss, loss_dict)
        
        Returns:
            total_loss: Weighted sum of part losses
            loss_dict (optional): Individual part losses for logging
        """
        B, T, D = pred.shape
        assert D == self.pose_dim, f"Expected pose_dim={self.pose_dim}, got {D}"
        
        # Compute base loss for each dimension
        base_loss = self._compute_loss(pred, target)  # [B, T, 214]
        
        # Split by body parts
        body_loss = base_loss[:, :, :self.body_dim]  # [B, T, 66]
        lhand_loss = base_loss[:, :, self.body_dim:self.body_dim + self.hand_dim]  # [B, T, 42]
        rhand_loss = base_loss[:, :, self.body_dim + self.hand_dim:self.body_dim + 2*self.hand_dim]  # [B, T, 42]
        face_loss = base_loss[:, :, self.body_dim + 2*self.hand_dim:]  # [B, T, 64]
        
        # Average over dimensions for each part
        body_loss_avg = body_loss.mean(dim=-1)  # [B, T]
        lhand_loss_avg = lhand_loss.mean(dim=-1)
        rhand_loss_avg = rhand_loss.mean(dim=-1)
        face_loss_avg = face_loss.mean(dim=-1)
        
        # Combine hand losses
        hands_loss_avg = (lhand_loss_avg + rhand_loss_avg) / 2  # [B, T]
        
        # Adaptive weighting (optional)
        if self.use_adaptive_weights:
            # Weight parts more if they're moving (high velocity)
            if T > 1:
                body_vel = torch.norm(pred[:, 1:, :self.body_dim] - pred[:, :-1, :self.body_dim], dim=-1).mean()
                hands_vel = torch.norm(
                    pred[:, 1:, self.body_dim:self.body_dim + 2*self.hand_dim] - 
                    pred[:, :-1, self.body_dim:self.body_dim + 2*self.hand_dim], 
                    dim=-1
                ).mean()
                face_vel = torch.norm(
                    pred[:, 1:, self.body_dim + 2*self.hand_dim:] - 
                    pred[:, :-1, self.body_dim + 2*self.hand_dim:], 
                    dim=-1
                ).mean()
                
                # Normalize velocities
                total_vel = body_vel + hands_vel + face_vel + 1e-6
                adaptive_body = (body_vel / total_vel) * 0.3 + 0.7 * self.weight_body
                adaptive_hands = (hands_vel / total_vel) * 0.3 + 0.7 * self.weight_hands
                adaptive_face = (face_vel / total_vel) * 0.3 + 0.7 * self.weight_face
                
                # Renormalize
                total_adaptive = adaptive_body + adaptive_hands + adaptive_face
                weight_body = adaptive_body / total_adaptive
                weight_hands = adaptive_hands / total_adaptive
                weight_face = adaptive_face / total_adaptive
            else:
                weight_body = self.weight_body
                weight_hands = self.weight_hands
                weight_face = self.weight_face
        else:
            weight_body = self.weight_body
            weight_hands = self.weight_hands
            weight_face = self.weight_face
        
        # Apply mask if provided
        if mask is not None:
            mask_float = mask.float()  # [B, T]
            body_loss_avg = body_loss_avg * mask_float
            hands_loss_avg = hands_loss_avg * mask_float
            face_loss_avg = face_loss_avg * mask_float
            
            n_valid = mask_float.sum().clamp(min=1.0)
            body_loss_scalar = body_loss_avg.sum() / n_valid
            hands_loss_scalar = hands_loss_avg.sum() / n_valid
            face_loss_scalar = face_loss_avg.sum() / n_valid
        else:
            body_loss_scalar = body_loss_avg.mean()
            hands_loss_scalar = hands_loss_avg.mean()
            face_loss_scalar = face_loss_avg.mean()
        
        # Weighted total
        total_loss = (
            weight_body * body_loss_scalar +
            weight_hands * hands_loss_scalar +
            weight_face * face_loss_scalar
        )
        
        if return_parts:
            loss_dict = {
                'body': body_loss_scalar.item(),
                'hands': hands_loss_scalar.item(),
                'face': face_loss_scalar.item(),
                'total': total_loss.item(),
                'weight_body': weight_body if isinstance(weight_body, float) else weight_body.item(),
                'weight_hands': weight_hands if isinstance(weight_hands, float) else weight_hands.item(),
                'weight_face': weight_face if isinstance(weight_face, float) else weight_face.item(),
            }
            return total_loss, loss_dict
        
        return total_loss


class SignLanguageLoss(nn.Module):
    """
    Complete Sign Language Loss with:
    - Weighted spatial loss (body/hands/face)
    - Temporal smoothness loss (velocity/acceleration)
    - Optional perceptual loss (future: VGG on pose rendering)
    
    Optimized for sign language where hands >>> body >>> face in importance.
    """
    
    def __init__(
        self,
        pose_dim: int = 534,
        body_dim: int = 24,
        hand_dim: int = 63,
        face_dim: int = 384,
        weight_body: float = 0.15,
        weight_hands: float = 0.55,
        weight_face: float = 0.30,
        weight_velocity: float = 0.1,
        weight_acceleration: float = 0.05,
        weight_bone: float = 0.1,  # Added bone length weight
        weight_bone_orientation: float = 0.1,  # ✅ Added bone orientation weight
        use_huber: bool = False,
        huber_delta: float = 1.0,  # ✅ Added huber_delta
        use_adaptive_weights: bool = False
    ):
        super().__init__()
        
        # Spatial weighted loss
        self.spatial_loss = PosePartWeightedLoss(
            pose_dim=pose_dim,
            body_dim=body_dim,
            hand_dim=hand_dim,
            face_dim=face_dim,
            weight_body=weight_body,
            weight_hands=weight_hands,
            weight_face=weight_face,
            use_huber=use_huber,
            huber_delta=huber_delta,  # ✅ Pass huber_delta
            use_adaptive_weights=use_adaptive_weights
        )
        
        # Temporal weights
        self.weight_velocity = weight_velocity
        self.weight_acceleration = weight_acceleration
        self.weight_bone = weight_bone
        self.weight_bone_orientation = weight_bone_orientation  # ✅ Store orientation weight
        
    def _compute_temporal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal consistency losses.
        
        Returns:
            velocity_loss, acceleration_loss
        """
        device = pred.device
        
        # Velocity loss
        if pred.shape[1] > 1:
            pred_vel = pred[:, 1:] - pred[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            vel_loss = (pred_vel - target_vel) ** 2
            
            if mask is not None:
                vel_mask = mask[:, 1:].unsqueeze(-1).float()
                vel_loss = (vel_loss * vel_mask).sum() / (vel_mask.sum() * pred.shape[-1]).clamp(min=1)
            else:
                vel_loss = vel_loss.mean()
        else:
            vel_loss = torch.tensor(0.0, device=device)
        
        # Acceleration loss
        if pred.shape[1] > 2:
            pred_accel = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
            target_accel = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
            accel_loss = (pred_accel - target_accel) ** 2
            
            if mask is not None:
                accel_mask = (mask[:, :-2] * mask[:, 1:-1] * mask[:, 2:]).unsqueeze(-1).float()
                accel_loss = (accel_loss * accel_mask).sum() / (accel_mask.sum() * pred.shape[-1]).clamp(min=1)
            else:
                accel_loss = accel_loss.mean()
        else:
            accel_loss = torch.tensor(0.0, device=device)
        
        return vel_loss, accel_loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_parts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute total sign language loss.
        
        Args:
            pred: [B, T, 214]
            target: [B, T, 214]
            mask: [B, T] - True = valid
            return_parts: If True, return (total_loss, loss_dict)
        
        Returns:
            total_loss or (total_loss, loss_dict)
        """
        # Spatial loss
        spatial_loss, spatial_dict = self.spatial_loss(pred, target, mask, return_parts=True)
        
        # Temporal losses
        vel_loss, accel_loss = self._compute_temporal_loss(pred, target, mask)
        
        # Bone length loss
        if self.weight_bone > 0:
            bone_loss = bone_length_loss(pred, target, mask)
        else:
            bone_loss = torch.tensor(0.0, device=pred.device)
            
        # Bone Orientation loss
        if self.weight_bone_orientation > 0:
            orient_loss = bone_orientation_loss(pred, target, mask)
        else:
            orient_loss = torch.tensor(0.0, device=pred.device)
        
        # Total
        total_loss = (
            spatial_loss +
            self.weight_velocity * vel_loss +
            self.weight_acceleration * accel_loss +
            self.weight_bone * bone_loss +
            self.weight_bone_orientation * orient_loss
        )
        
        if return_parts:
            loss_dict = {
                **spatial_dict,
                'velocity': vel_loss.item(),
                'acceleration': accel_loss.item(),
                'bone': bone_loss.item(),  # Add bone loss
                'orientation': orient_loss.item(),  # ✅ Add orientation loss
                'total_with_temporal': total_loss.item()
            }
            return total_loss, loss_dict
        
        return total_loss


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Weighted Pose Loss")
    print("=" * 60)
    
    # Test 1: Basic forward pass
    print("\n1. Testing PosePartWeightedLoss...")
    loss_fn = PosePartWeightedLoss(
        weight_hands=0.55,
        weight_face=0.30,
        weight_body=0.15
    )
    
    pred = torch.randn(2, 50, 534)
    target = torch.randn(2, 50, 534)
    mask = torch.ones(2, 50).bool()
    
    loss, loss_dict = loss_fn(pred, target, mask, return_parts=True)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Body: {loss_dict['body']:.4f} (weight: {loss_dict['weight_body']:.2f})")
    print(f"   Hands: {loss_dict['hands']:.4f} (weight: {loss_dict['weight_hands']:.2f})")
    print(f"   Face: {loss_dict['face']:.4f} (weight: {loss_dict['weight_face']:.2f})")
    print("   ✅ PASS")
    
    # Test 2: Verify hands get higher weight
    print("\n2. Testing hand priority...")
    pred_perfect_body = target.clone()
    pred_perfect_body[:, :, 24:] += 1.0  # Corrupt hands and face
    
    pred_perfect_hands = target.clone()
    pred_perfect_hands[:, :, :24] += 1.0  # Corrupt body
    
    # pred_perfect_body: Body is Good, Hands/Face are Bad (Loss should be HIGH)
    # pred_perfect_hands: Hands/Face are Good, Body is Bad (Loss should be LOW)
    
    loss_when_hands_bad, _ = loss_fn(pred_perfect_body, target, mask, return_parts=True)
    loss_when_body_bad, _ = loss_fn(pred_perfect_hands, target, mask, return_parts=True)
    
    print(f"   Loss (Hands Bad): {loss_when_hands_bad.item():.4f}")
    print(f"   Loss (Body Bad) : {loss_when_body_bad.item():.4f}")
    assert loss_when_hands_bad > loss_when_body_bad, "Hands should contribute more to loss!"
    print("   ✅ PASS - Hands prioritized correctly")
    
    # Test 3: SignLanguageLoss with temporal
    print("\n3. Testing SignLanguageLoss (with temporal)...")
    sl_loss = SignLanguageLoss(
        weight_hands=0.55,
        weight_velocity=0.1,
        weight_acceleration=0.05
    )
    
    total_loss, full_dict = sl_loss(pred, target, mask, return_parts=True)
    print(f"   Total: {total_loss.item():.4f}")
    print(f"   Spatial: {full_dict['total']:.4f}")
    print(f"   Velocity: {full_dict['velocity']:.4f}")
    print(f"   Acceleration: {full_dict['acceleration']:.4f}")
    print("   ✅ PASS")

    # Test 4: Bone Length Loss
    print("\n4. Testing Bone Length Loss...")
    sl_loss_with_bone = SignLanguageLoss(
        weight_bone=0.1
    )
    
    # Create stretch artifact: Scale one skeleton by 1.5x (bones will stretch)
    pred_stretched = target.clone() * 1.5
    
    loss, loss_dict = sl_loss_with_bone(pred_stretched, target, mask, return_parts=True)
    print(f"   Bone Loss: {loss_dict['bone']:.4f}")
    assert loss_dict['bone'] > 0.001, "Bone loss should be significant!"
    print("   ✅ PASS - Bone loss active")
    
    # Test 5: Adaptive weighting
    print("\n4. Testing adaptive weights...")
    adaptive_loss = PosePartWeightedLoss(
        weight_hands=0.55,
        use_adaptive_weights=True
    )
    
    # Create sequence with hand motion
    pred_moving_hands = target.clone()
    # Hands: Body(24) -> End of Hands(24+63+63=150)
    pred_moving_hands[:, :, 24:150] += torch.linspace(0, 2, 50).view(1, 50, 1)
    
    loss_adaptive, dict_adaptive = adaptive_loss(pred_moving_hands, target, mask, return_parts=True)
    print(f"   Adaptive hands weight: {dict_adaptive['weight_hands']:.3f}")
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
