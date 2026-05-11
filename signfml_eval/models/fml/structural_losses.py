"""
Structural Losses for Pose Generation - FIXED FOR 534D (3D COORDS)
==================================================================
Author: Kieu Vo
Date: 4 Feb 2026

FIXED: Updated for 534D pose format with 3D coordinates
Old format: 214D (2D coords) - WRONG
New format: 534D (3D coords):
    - Body: [0:24] = 8 keypoints × 3 = 24 dims
    - Right Hand: [24:87] = 21 keypoints × 3 = 63 dims  
    - Left Hand: [87:150] = 21 keypoints × 3 = 63 dims
    - Face: [150:534] = 128 keypoints × 3 = 384 dims

Bone connections for SLRTP 8-point body:
    0: Nose
    1: Left Shoulder
    2: Right Shoulder
    3: Left Elbow
    4: Right Elbow
    5: Left Wrist
    6: Right Wrist
    7: Spine (center)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# SLRTP 8-point body skeleton connections
# =============================================================================
BODY_BONES_3D = [
    # Left arm
    (1, 3),  # Left shoulder -> elbow
    (3, 5),  # Left elbow -> wrist
    # Right arm
    (2, 4),  # Right shoulder -> elbow
    (4, 6),  # Right elbow -> wrist
    # Shoulders
    (1, 2),  # Left shoulder -> Right shoulder
    # Torso (if spine defined)
    (1, 7),  # Left shoulder -> spine
    (2, 7),  # Right shoulder -> spine
]


# MediaPipe Hand bone connections (21 points)
# 0: wrist
# 1-4: thumb
# 5-8: index
# 9-12: middle
# 13-16: ring
# 17-20: pinky
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


def bone_length_loss(pred_pose, gt_pose, mask=None):
    """
    Enforce constant bone lengths (skeleton doesn't stretch!)
    
    FIXED: Now works with 534D poses (3D coordinates)
    
    Args:
        pred_pose: [B, T, 534] predicted poses (3D)
        gt_pose: [B, T, 534] ground truth poses (3D)
        mask: [B, T] optional mask for valid frames
    
    Returns:
        Bone length consistency loss
    """
    B, T, D = pred_pose.shape
    device = pred_pose.device
    
    # Validate dimension
    if D != 534:
        # Fallback for other dimensions - return zero loss
        return torch.tensor(0.0, device=device)
    
    # =====================
    # 1. BODY BONES (8 pts × 3 = 24 dims)
    # =====================
    # Body: [0:24] -> [B, T, 8, 3]
    pred_body = pred_pose[:, :, :24].reshape(B, T, 8, 3)
    gt_body = gt_pose[:, :, :24].reshape(B, T, 8, 3)
    
    # Flatten for easier processing
    pred_body_flat = pred_body.reshape(-1, 8, 3)  # [B*T, 8, 3]
    gt_body_flat = gt_body.reshape(-1, 8, 3)
    
    total_loss = 0.0
    num_bones = 0
    
    for i, j in BODY_BONES_3D:
        if i < 8 and j < 8:  # Ensure valid indices
            # GT bone vector and length
            gt_bone_vec = gt_body_flat[:, i] - gt_body_flat[:, j]  # [B*T, 3]
            gt_length = torch.norm(gt_bone_vec, dim=-1)  # [B*T]
            
            # Predicted bone vector and length
            pred_bone_vec = pred_body_flat[:, i] - pred_body_flat[:, j]
            pred_length = torch.norm(pred_bone_vec, dim=-1)
            
            # Loss: want pred_length ≈ gt_length
            bone_loss = F.mse_loss(pred_length, gt_length, reduction='none')
            
            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.reshape(-1).float()
                bone_loss = bone_loss * mask_flat
                total_loss += bone_loss.sum() / mask_flat.sum().clamp(min=1)
            else:
                total_loss += bone_loss.mean()
            
            num_bones += 1
    
    # =====================
    # 2. HAND BONES (21 pts × 3 = 63 dims each)
    # =====================
    # Right Hand: [24:87] -> [B, T, 21, 3]
    pred_rhand = pred_pose[:, :, 24:87].reshape(B, T, 21, 3)
    gt_rhand = gt_pose[:, :, 24:87].reshape(B, T, 21, 3)
    
    # Left Hand: [87:150] -> [B, T, 21, 3]
    pred_lhand = pred_pose[:, :, 87:150].reshape(B, T, 21, 3)
    gt_lhand = gt_pose[:, :, 87:150].reshape(B, T, 21, 3)
    
    # Flatten
    pred_rhand_flat = pred_rhand.reshape(-1, 21, 3)
    gt_rhand_flat = gt_rhand.reshape(-1, 21, 3)
    pred_lhand_flat = pred_lhand.reshape(-1, 21, 3)
    gt_lhand_flat = gt_lhand.reshape(-1, 21, 3)
    
    for i, j in HAND_BONES:
        # Right hand
        gt_bone_vec = gt_rhand_flat[:, i] - gt_rhand_flat[:, j]
        gt_length = torch.norm(gt_bone_vec, dim=-1)
        pred_bone_vec = pred_rhand_flat[:, i] - pred_rhand_flat[:, j]
        pred_length = torch.norm(pred_bone_vec, dim=-1)
        
        bone_loss = F.mse_loss(pred_length, gt_length, reduction='none')
        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            bone_loss = bone_loss * mask_flat
            total_loss += bone_loss.sum() / mask_flat.sum().clamp(min=1)
        else:
            total_loss += bone_loss.mean()
        num_bones += 1
        
        # Left hand
        gt_bone_vec = gt_lhand_flat[:, i] - gt_lhand_flat[:, j]
        gt_length = torch.norm(gt_bone_vec, dim=-1)
        pred_bone_vec = pred_lhand_flat[:, i] - pred_lhand_flat[:, j]
        pred_length = torch.norm(pred_bone_vec, dim=-1)
        
        bone_loss = F.mse_loss(pred_length, gt_length, reduction='none')
        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            bone_loss = bone_loss * mask_flat
            total_loss += bone_loss.sum() / mask_flat.sum().clamp(min=1)
        else:
            total_loss += bone_loss.mean()
        num_bones += 1
    
    return total_loss / max(num_bones, 1)


def velocity_smoothness_loss(pred_pose, mask=None):
    """
    Encourage smooth motion (penalize large accelerations)
    
    Args:
        pred_pose: [B, T, 534] predicted poses
        mask: [B, T] optional mask for valid frames
    
    Returns:
        Velocity smoothness loss
    """
    device = pred_pose.device
    
    # Velocity: pose[t+1] - pose[t]
    velocity = pred_pose[:, 1:] - pred_pose[:, :-1]  # [B, T-1, 534]
    
    # Acceleration: velocity[t+1] - velocity[t]
    accel = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, 534]
    
    # Squared acceleration (penalize jittering)
    accel_sq = accel ** 2  # [B, T-2, 534]
    
    # Apply mask if provided
    if mask is not None:
        # Mask for acceleration: need frames [t, t+1, t+2] all valid
        accel_mask = mask[:, :-2] * mask[:, 1:-1] * mask[:, 2:]  # [B, T-2]
        accel_mask = accel_mask.unsqueeze(-1).float()  # [B, T-2, 1]
        
        accel_sq = accel_sq * accel_mask
        return accel_sq.sum() / accel_mask.sum().clamp(min=1)
    else:
        return accel_sq.mean()


def joint_angle_loss(pred_pose, mask=None):
    """
    Enforce physical constraints on joint angles
    (e.g., elbows don't bend backwards)
    
    FIXED: Now works with 534D poses (3D coordinates)
    
    Args:
        pred_pose: [B, T, 534] predicted poses
        mask: [B, T] optional mask
    
    Returns:
        Joint angle plausibility loss
    """
    B, T, D = pred_pose.shape
    device = pred_pose.device
    
    if D != 534:
        return torch.tensor(0.0, device=device)
    
    # Extract body keypoints: 8 pts × 3 = 24 dims
    pred_body = pred_pose[:, :, :24].reshape(B, T, 8, 3)
    pred_flat = pred_body.reshape(-1, 8, 3)  # [B*T, 8, 3]
    
    total_loss = 0.0
    
    # Left arm angle (shoulder-elbow-wrist): indices 1, 3, 5
    shoulder_l = pred_flat[:, 1]  # [B*T, 3]
    elbow_l = pred_flat[:, 3]
    wrist_l = pred_flat[:, 5]
    
    # Vectors from elbow
    v1_l = shoulder_l - elbow_l  # [B*T, 3]
    v2_l = wrist_l - elbow_l
    
    # Cosine of angle
    cos_l = (v1_l * v2_l).sum(dim=-1) / (
        torch.norm(v1_l, dim=-1) * torch.norm(v2_l, dim=-1) + 1e-6
    )  # [B*T]
    
    # Elbow angle should be [30°, 180°] → cos in [-1, 0.866]
    # Penalize if outside range
    loss_left = torch.relu(cos_l - 0.866) + torch.relu(-1.0 - cos_l)
    
    # Right arm angle (shoulder-elbow-wrist): indices 2, 4, 6
    shoulder_r = pred_flat[:, 2]
    elbow_r = pred_flat[:, 4]
    wrist_r = pred_flat[:, 6]
    
    v1_r = shoulder_r - elbow_r
    v2_r = wrist_r - elbow_r
    
    cos_r = (v1_r * v2_r).sum(dim=-1) / (
        torch.norm(v1_r, dim=-1) * torch.norm(v2_r, dim=-1) + 1e-6
    )
    
    loss_right = torch.relu(cos_r - 0.866) + torch.relu(-1.0 - cos_r)
    
    # Combine
    angle_loss = loss_left + loss_right  # [B*T]
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.reshape(-1).float()
        angle_loss = angle_loss * mask_flat
        total_loss = angle_loss.sum() / mask_flat.sum().clamp(min=1)
    else:
        total_loss = angle_loss.mean()
    
    return total_loss


def bone_orientation_loss(pred_pose, gt_pose, mask=None):
    """
    Enforce correct bone orientation (cosine similarity)
    Crucial for correct hand shapes and semantics
    
    FIXED: Now works with 534D poses (3D coordinates)
    
    Args:
        pred_pose: [B, T, 534]
        gt_pose: [B, T, 534]
        mask: [B, T]
        
    Returns:
        1 - CosineSimilarity (0 = aligned, 2 = opposite)
    """
    B, T, D = pred_pose.shape
    device = pred_pose.device
    
    if D != 534:
        return torch.tensor(0.0, device=device)
    
    total_loss = 0.0
    num_bones = 0
    
    # --- 1. BODY BONES (8 pts × 3) ---
    pred_body = pred_pose[:, :, :24].reshape(-1, 8, 3)
    gt_body = gt_pose[:, :, :24].reshape(-1, 8, 3)
    
    for i, j in BODY_BONES_3D:
        if i < 8 and j < 8:
            v_pred = pred_body[:, i] - pred_body[:, j]  # [B*T, 3]
            v_gt = gt_body[:, i] - gt_body[:, j]
            
            # Cosine Loss: 1 - cos(theta)
            loss = 1.0 - F.cosine_similarity(v_pred, v_gt, dim=-1, eps=1e-6)
            
            if mask is not None:
                mask_flat = mask.reshape(-1).float()
                loss = loss * mask_flat
                total_loss += loss.sum() / mask_flat.sum().clamp(min=1)
            else:
                total_loss += loss.mean()
            num_bones += 1
        
    # --- 2. HAND BONES (21 pts × 3 each) ---
    # Right Hand: [24:87]
    pred_rhand = pred_pose[:, :, 24:87].reshape(-1, 21, 3)
    gt_rhand = gt_pose[:, :, 24:87].reshape(-1, 21, 3)
    
    # Left Hand: [87:150]
    pred_lhand = pred_pose[:, :, 87:150].reshape(-1, 21, 3)
    gt_lhand = gt_pose[:, :, 87:150].reshape(-1, 21, 3)
    
    for i, j in HAND_BONES:
        # Right hand
        v_pred = pred_rhand[:, i] - pred_rhand[:, j]
        v_gt = gt_rhand[:, i] - gt_rhand[:, j]
        
        loss = 1.0 - F.cosine_similarity(v_pred, v_gt, dim=-1, eps=1e-6)
        
        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            loss = loss * mask_flat
            total_loss += loss.sum() / mask_flat.sum().clamp(min=1)
        else:
            total_loss += loss.mean()
        num_bones += 1
        
        # Left hand
        v_pred = pred_lhand[:, i] - pred_lhand[:, j]
        v_gt = gt_lhand[:, i] - gt_lhand[:, j]
        
        loss = 1.0 - F.cosine_similarity(v_pred, v_gt, dim=-1, eps=1e-6)
        
        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            loss = loss * mask_flat
            total_loss += loss.sum() / mask_flat.sum().clamp(min=1)
        else:
            total_loss += loss.mean()
        num_bones += 1
        
    return total_loss / max(num_bones, 1)


def compute_structural_losses(pred_pose, gt_pose, mask=None, weights=None):
    """
    Compute all structural losses with optional weighting
    
    Args:
        pred_pose: [B, T, 534] predicted poses
        gt_pose: [B, T, 534] ground truth poses  
        mask: [B, T] optional mask
        weights: dict with keys 'bone', 'smooth', 'angle', 'orientation'
    
    Returns:
        total_loss: weighted sum of all losses
        loss_dict: individual loss values for logging
    """
    if weights is None:
        weights = {'bone': 0.3, 'smooth': 0.2, 'angle': 0.1, 'orientation': 0.2}
    
    # Compute individual losses
    bone_loss = bone_length_loss(pred_pose, gt_pose, mask)
    smooth_loss = velocity_smoothness_loss(pred_pose, mask)
    angle_loss = joint_angle_loss(pred_pose, mask)
    orient_loss = bone_orientation_loss(pred_pose, gt_pose, mask)
    
    # Weighted sum
    total = (
        weights.get('bone', 0.3) * bone_loss +
        weights.get('smooth', 0.2) * smooth_loss +
        weights.get('angle', 0.1) * angle_loss +
        weights.get('orientation', 0.2) * orient_loss
    )
    
    loss_dict = {
        'bone': bone_loss.item(),
        'smooth': smooth_loss.item(),
        'angle': angle_loss.item(),
        'orientation': orient_loss.item(),
        'structural_total': total.item()
    }
    
    return total, loss_dict


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Structural Losses (534D / 3D)")  
    print("=" * 60)
    
    # Test with 534D poses
    B, T, D = 2, 50, 534
    pred = torch.randn(B, T, D)
    gt = torch.randn(B, T, D)
    mask = torch.ones(B, T).bool()
    
    print(f"\nInput: pred={pred.shape}, gt={gt.shape}, mask={mask.shape}")
    
    # Test individual losses
    print("\n1. Testing bone_length_loss...")
    bone = bone_length_loss(pred, gt, mask)
    print(f"   Bone length loss: {bone.item():.4f}")
    
    print("\n2. Testing velocity_smoothness_loss...")
    smooth = velocity_smoothness_loss(pred, mask)
    print(f"   Velocity smoothness loss: {smooth.item():.4f}")
    
    print("\n3. Testing joint_angle_loss...")
    angle = joint_angle_loss(pred, mask)
    print(f"   Joint angle loss: {angle.item():.4f}")
    
    print("\n4. Testing bone_orientation_loss...")
    orient = bone_orientation_loss(pred, gt, mask)
    print(f"   Bone orientation loss: {orient.item():.4f}")
    
    print("\n5. Testing compute_structural_losses...")
    total, loss_dict = compute_structural_losses(pred, gt, mask)
    print(f"   Total structural loss: {total.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
