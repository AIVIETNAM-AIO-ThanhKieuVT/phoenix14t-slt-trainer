"""
✅ FINAL FIX: Mask convention chuẩn hóa HOÀN TOÀN
Convention: True = VALID, False = PADDING (consistent everywhere)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
try:
    from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
    from .transformer_prior import TransformerPrior
    from .sync_guidance import SyncGuidanceHead 
except ImportError:
    print("⚠️ Warning: External modules not found.")
    FlowMatchingBlock = None
    FlowMatchingScheduler = None
    FlowMatchingLoss = None
    TransformerPrior = None
    SyncGuidanceHead = None
class LengthPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    def forward(self, text_features, mask=None):
        """
        Args:
            text_features: [B, L, D]
            mask: [B, L] - ✅ True = VALID, False = PADDING
        """
        # ✅ Input validation
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            print(f"⚠️ LengthPredictor: NaN/Inf in text_features input!")
            text_features = torch.nan_to_num(text_features, nan=0.0, posinf=5.0, neginf=-5.0)
        
        text_features = torch.clamp(text_features, -10.0, 10.0)
        
        if mask is not None:
            valid_mask = mask.float().unsqueeze(-1)  # [B, L, 1]
            sum_features = (text_features * valid_mask).sum(dim=1)
            # Safe division with larger epsilon
            sum_counts = valid_mask.sum(dim=1).clamp(min=1.0)  # At least 1
            pooled_text = sum_features / sum_counts
        else:
            pooled_text = text_features.mean(dim=1)
        
        # Clamp pooled features
        pooled_text = torch.clamp(pooled_text, -10.0, 10.0)
            
        pred_length = self.net(pooled_text)
        
        # ✅ CRITICAL: Clamp output to reasonable range
        pred_length = torch.clamp(pred_length, min=1.0, max=500.0)
        
        # ✅ Check for NaN
        if torch.isnan(pred_length).any() or torch.isinf(pred_length).any():
            print(f"⚠️ LengthProdictor: NaN/Inf in output! Replacing with safe values.")
            pred_length = torch.nan_to_num(pred_length, nan=50.0, posinf=500.0, neginf=1.0)
        
        return pred_length.squeeze(-1)
# ...
    def _train_forward(self, batch, text_features, text_mask, gt_latent,
                      pose_gt=None, return_attn_weights=False, prior_scale=1.0):
        """Training forward pass"""
        device = text_features.device
        B, T, D = gt_latent.shape
        
        # Loss 1: Length prediction
        pred_length = self.length_predictor(text_features.detach(), text_mask)
        # Add slight epsilon to avoid extremely small values if LENGTH_SCALE is huge
        length_loss = self.length_loss_fn(
            pred_length / self.LENGTH_SCALE,
            batch['seq_lengths'].float() / self.LENGTH_SCALE
        )
        
        # Prepare masks (True=valid, False=padding)
        t = self.scheduler.sample_timesteps(B, device)
        valid_mask = torch.arange(T, device=device)[None, :] < batch['seq_lengths'][:, None]
        
        # For Transformer: invert masks
        text_padding_mask = ~text_mask
        pose_padding_mask = ~valid_mask
        
        # Add noise
        latent_t, v_gt, _ = self.scheduler.add_noise(gt_latent, t)
        
        # Loss 2: Prior matching
        v_prior = None
        prior_loss = torch.tensor(0.0, device=device)
        if self.use_ssm_prior and self.ssm_prior is not None:
            v_prior = self.ssm_prior(latent_t, t, text_features, mask=valid_mask)
            # Numerical stability for loss
            denom = (valid_mask.sum() * D).clamp(min=1.0)
            prior_loss = ((v_prior - v_gt.detach())**2 * valid_mask.unsqueeze(-1)).sum() / denom
        
        # Main flow prediction
        out = self.flow_block(
            latent_t, t, text_features,
            text_attn_mask=text_padding_mask,
            pose_attn_mask=pose_padding_mask,
            return_attn=return_attn_weights
        )
        v_flow = out[0] if return_attn_weights else out
        
        # Target velocity
        lambda_t = self.get_lambda_t(t) * prior_scale
        v_target = v_gt - lambda_t * v_prior.detach() if v_prior is not None else v_gt
        
        # Loss 3: Sync guidance
        sync_loss = torch.tensor(0.0, device=device)
        if self.use_sync_guidance and self.sync_head is not None:
            text_valid_mask = text_mask.unsqueeze(-1).float()
            # Safe pool
            text_denom = text_valid_mask.sum(1).clamp(min=1e-6)
            text_pooled = (text_features * text_valid_mask).sum(1) / text_denom
            
            # Safe average over time
            valid_mask_f = valid_mask.float()
            time_denom = valid_mask_f.sum(1).clamp(min=1e-6)
            
            pos = (self.sync_head(latent_t, text_pooled, valid_mask) * valid_mask_f).sum(1) / time_denom
            
            neg = (self.sync_head(latent_t, text_pooled[torch.randperm(B, device=device)], valid_mask) * 
                   valid_mask_f).sum(1) / time_denom
            
            sync_loss = F.relu(0.5 + neg - pos).mean()
        
        # Loss 4: Flow matching
        flow_loss = self.flow_loss_fn(v_flow, v_target, mask=valid_mask)
        
        # Total loss
        total_loss = flow_loss + \
                    self.W_PRIOR * prior_loss + \
                    self.W_SYNC * sync_loss + \
                    self.W_LENGTH * length_loss
        
        v_pred_train = v_flow + lambda_t * v_prior.detach() if v_prior is not None else v_flow
        
        result = {
            'total': total_loss,
            'flow': flow_loss,
            'prior': prior_loss,
            'sync': sync_loss,
            'length': length_loss,
            'predicted_latent': latent_t.detach(),
            'velocity_pred': v_pred_train.detach(),
            'velocity_target': v_target.detach(),
            'v_flow': v_flow.detach(),
            'v_prior': v_prior.detach() if v_prior is not None else None,
            'timestep': t.detach(),
            'valid_mask': valid_mask.detach()
        }
        
        if return_attn_weights:
            result['attn_weights'] = out[1]
        
        return result
class LatentFlowMatcher(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        text_encoder_name='bert-base-multilingual-cased',
        hidden_dim=512,
        num_flow_layers=6,
        num_prior_layers=4,
        num_heads=8,
        dropout=0.1,
        use_ssm_prior=True,
        use_sync_guidance=True,
        lambda_prior=0.1,
        gamma_guidance=0.01,
        lambda_anneal=True,
        W_PRIOR=0.1,
        W_SYNC=0.1,
        W_LENGTH=1.0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_ssm_prior = use_ssm_prior
        self.use_sync_guidance = use_sync_guidance
        self.lambda_prior = lambda_prior
        self.gamma_guidance = gamma_guidance
        self.lambda_anneal = lambda_anneal
        
        self.W_PRIOR = W_PRIOR
        self.W_SYNC = W_SYNC
        self.W_LENGTH = W_LENGTH
        self.LENGTH_SCALE = 120.0
        # Text Encoder (Frozen BERT)
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Length Predictor
        self.length_predictor = LengthPredictor(input_dim=hidden_dim)
        self.length_loss_fn = nn.SmoothL1Loss()
        
        # Flow Matching
        self.scheduler = FlowMatchingScheduler() if FlowMatchingScheduler else None
        self.flow_block = FlowMatchingBlock(
            data_dim=latent_dim,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        ) if FlowMatchingBlock else None
        
        # Optional modules
        if use_ssm_prior and TransformerPrior:
            self.ssm_prior = TransformerPrior(
                latent_dim, hidden_dim, num_prior_layers, num_heads, dropout
            )
        else:
            self.ssm_prior = None
        if use_sync_guidance and SyncGuidanceHead:
            self.sync_head = SyncGuidanceHead(
                latent_dim=latent_dim, 
                hidden_dim=hidden_dim // 2, 
                dropout=dropout,
                text_dim=hidden_dim  # <--- THÊM DÒNG NÀY
            )
        else:
            self.sync_head = None
        
        self.flow_loss_fn = FlowMatchingLoss() if FlowMatchingLoss else None
    def encode_text(self, text_tokens, attention_mask):
        """
        Encode text with BERT
        
        Args:
            text_tokens: [B, L]
            attention_mask: [B, L] - BERT convention: 1=valid, 0=padding
        
        Returns:
            text_features: [B, L, hidden_dim]
            text_mask: [B, L] - ✅ True=VALID, False=PADDING
        """
        self.text_encoder.eval()
        with torch.no_grad():
            outputs = self.text_encoder(text_tokens, attention_mask=attention_mask)
        
        text_features = self.text_proj(outputs.last_hidden_state)
        
        # ✅ Convert BERT mask (1=valid) to boolean (True=valid)
        text_mask = attention_mask.bool()  # True=valid, False=padding
        
        return text_features, text_mask
    
    def get_lambda_t(self, t):
        """
        Get prior weighting at timestep t
        
        Args:
            t: [B] or scalar tensor
        
        Returns:
            lambda_t: [B, 1, 1]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        batch_size = t.shape[0]
        
        if self.lambda_anneal:
            lambda_val = self.lambda_prior * (1 - t.float())
            return lambda_val.view(batch_size, 1, 1)
        else:
            return torch.full(
                (batch_size, 1, 1),
                self.lambda_prior,
                device=t.device,
                dtype=torch.float32
            )
    def forward(self, batch, gt_latent, pose_gt=None, mode='train',
                num_inference_steps=50, return_attn_weights=False, prior_scale=1.0):
        """Forward pass"""
        text_features, text_mask = self.encode_text(
            batch['text_tokens'],
            batch['attention_mask']
        )
        
        if mode == 'train':
            return self._train_forward(
                batch, text_features, text_mask, gt_latent,
                pose_gt, return_attn_weights, prior_scale
            )
        else:
            return self.sample(
                batch, steps=num_inference_steps, device=text_features.device,
                text_features=text_features, text_mask=text_mask
            )
    
    @torch.no_grad()
    def sample(self, batch, steps=50, device='cuda', temperature=1.0,
               text_features=None, text_mask=None, max_seq_len=400,
               latent_scale=1.0, cfg_scale=1.5):
        """
        Inference sampling with Classifier-Free Guidance (CFG)
        
        Args:
            cfg_scale: Guidance scale for CFG (1.0 = no guidance, 1.5-2.0 recommended)
            latent_scale: Scale factor used during training
        """
        if text_features is None:
            text_features, text_mask = self.encode_text(
                batch['text_tokens'],
                batch['attention_mask']
            )
        
        B = text_features.shape[0]
        
        # 1. Predict length
        pred_len_raw = self.length_predictor(text_features, text_mask)
        seq_lens = pred_len_raw.round().long().clamp(min=10, max=max_seq_len)
        T = seq_lens.max().item()
        
        # 2. Create masks (True=valid, False=padding)
        valid_mask = torch.arange(T, device=device)[None, :] < seq_lens[:, None]
        
        # 3. Initialize noise
        z = torch.randn(B, T, self.latent_dim, device=device) * temperature
        
        # ✅ Prepare null embeddings for CFG (unconditional)
        use_cfg = cfg_scale != 1.0
        null_text_features = None
        null_text_mask = None
        if use_cfg:
            null_text_features = torch.zeros_like(text_features)
            null_text_mask = torch.zeros_like(text_mask)
        
        # 4. Euler integration with CFG
        dt = 1.0 / steps
        for step in range(steps):
            t_val = step / steps
            t = torch.full((B,), t_val, device=device)
            
            # Conditional velocity
            v_cond = self._compute_velocity(
                z, t, text_features, text_mask, valid_mask
            )
            
            # Apply CFG if enabled
            if use_cfg:
                # Unconditional velocity (null text)
                v_uncond = self._compute_velocity(
                    z, t, null_text_features, null_text_mask, valid_mask
                )
                
                # Guided velocity: v = v_uncond + cfg_scale * (v_cond - v_uncond)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond
            
            z = z + v_pred * dt
            z = z * valid_mask.unsqueeze(-1).float()
        
        # Un-scale
        if latent_scale != 1.0:
            z = z / latent_scale
            
        return z
    def _compute_velocity(self, latent, t, text_features, text_mask, valid_mask):
        """
        Compute velocity field
        
        Args:
            latent: [B, T, D]
            t: [B] timesteps
            text_features: [B, L, hidden_dim]
            text_mask: [B, L] - True=valid, False=padding
            valid_mask: [B, T] - True=valid, False=padding
        """
        # ✅ For Transformer: need INVERTED mask (True=ignore)
        text_padding_mask = ~text_mask  # True=padding (for attention mask)
        pose_padding_mask = ~valid_mask  # True=padding (for attention mask)
        
        # 1. Main flow
        v_flow = self.flow_block(
            latent, t, text_features,
            text_attn_mask=text_padding_mask,
            pose_attn_mask=pose_padding_mask
        )
        
        # 2. Prior (SSM)
        if self.use_ssm_prior and self.ssm_prior is not None:
            # ✅ SSM expects True=valid
            v_prior = self.ssm_prior(latent, t, text_features, mask=valid_mask)
            lambda_t = self.get_lambda_t(t)
            v_flow = v_flow + lambda_t * v_prior
        
        # 3. Sync guidance
        t_val = t[0].item() if t.numel() > 0 else 0.5
        
        if self.use_sync_guidance and self.sync_head is not None and 0.2 < t_val < 0.8:
            with torch.enable_grad():
                latent_grad = latent.detach().requires_grad_(True)
                
                # Pool text (using valid mask)
                text_valid_mask = text_mask.unsqueeze(-1).float()
                text_pooled = (text_features * text_valid_mask).sum(1) / \
                              text_valid_mask.sum(1).clamp(min=1)
                
                # Sync score (using valid mask)
                sync_score = self.sync_head(latent_grad, text_pooled, valid_mask)
                total_score = (sync_score * valid_mask.float()).sum() / \
                              valid_mask.sum().clamp(min=1)
                
                sync_grad = torch.autograd.grad(total_score, latent_grad)[0]
            
            # Clip gradient
            v_mag = v_flow.norm(dim=-1, keepdim=True).mean()
            clamp_val = min(0.1, max(0.01, v_mag.item() * 0.1))
            sync_grad = torch.clamp(sync_grad, -clamp_val, clamp_val)
            
            v_flow = v_flow + self.gamma_guidance * sync_grad.detach() * \
                     valid_mask.unsqueeze(-1).float()
        
        return v_flow
    def _train_forward(self, batch, text_features, text_mask, gt_latent,
                      pose_gt=None, return_attn_weights=False, prior_scale=1.0):
        """Training forward pass"""
        device = text_features.device
        B, T, D = gt_latent.shape
        
        # Loss 1: Length prediction
        pred_length = self.length_predictor(text_features.detach(), text_mask)
        length_loss = self.length_loss_fn(
            pred_length / self.LENGTH_SCALE,
            batch['seq_lengths'].float() / self.LENGTH_SCALE
        )
        
        # Prepare masks (True=valid, False=padding)
        t = self.scheduler.sample_timesteps(B, device)
        valid_mask = torch.arange(T, device=device)[None, :] < batch['seq_lengths'][:, None]
        
        # For Transformer: invert masks
        text_padding_mask = ~text_mask
        pose_padding_mask = ~valid_mask
        
        # Add noise
        latent_t, v_gt, _ = self.scheduler.add_noise(gt_latent, t)
        
        # ✅ DEBUG: Check noise result
        if torch.isnan(latent_t).any() or torch.isinf(latent_t).any():
            print(f"⚠️ NaN/Inf in latent_t after add_noise!")
            print(f"   gt_latent has NaN: {torch.isnan(gt_latent).any()}")
            print(f"   t: {t}")
            # Force to valid values
            latent_t = torch.nan_to_num(latent_t, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(v_gt).any() or torch.isinf(v_gt).any():
            print(f"⚠️ NaN/Inf in v_gt after add_noise!")
            v_gt = torch.nan_to_num(v_gt, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Loss 2: Prior matching
        v_prior = None
        prior_loss = torch.tensor(0.0, device=device)
        if self.use_ssm_prior and self.ssm_prior is not None:
            v_prior = self.ssm_prior(latent_t, t, text_features, mask=valid_mask)
            
            # ✅ DEBUG: Check prior output
            if torch.isnan(v_prior).any() or torch.isinf(v_prior).any():
                print(f"⚠️ NaN/Inf in v_prior from ssm_prior!")
                v_prior = torch.nan_to_num(v_prior, nan=0.0, posinf=1.0, neginf=-1.0)
            
            prior_loss = ((v_prior - v_gt.detach())**2 * valid_mask.unsqueeze(-1)).sum() / \
                        (valid_mask.sum() * D).clamp(min=1)
        
        # Main flow prediction
        out = self.flow_block(
            latent_t, t, text_features,
            text_attn_mask=text_padding_mask,
            pose_attn_mask=pose_padding_mask,
            return_attn=return_attn_weights
        )
        v_flow = out[0] if return_attn_weights else out
        
        # ✅ DEBUG: Check flow_block output
        if torch.isnan(v_flow).any() or torch.isinf(v_flow).any():
            print(f"⚠️ NaN/Inf in v_flow from flow_block!")
            print(f"   Percentage NaN: {torch.isnan(v_flow).float().mean()*100:.1f}%")
            print(f"   latent_t had NaN before: {torch.isnan(latent_t).any()}")
            v_flow = torch.nan_to_num(v_flow, nan=0.0, posinf=5.0, neginf=-5.0)
        
        v_flow = v_flow.clamp(-10.0, 10.0)
        
        # Target velocity
        lambda_t = self.get_lambda_t(t) * prior_scale
        v_target = v_gt - lambda_t * v_prior.detach() if v_prior is not None else v_gt
        
        # ✅ DEBUG: Check target
        if torch.isnan(v_target).any() or torch.isinf(v_target).any():
            print(f"⚠️ NaN/Inf in v_target!")
            v_target = torch.nan_to_num(v_target, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Loss 3: Sync guidance
        sync_loss = torch.tensor(0.0, device=device)
        if self.use_sync_guidance and self.sync_head is not None:
            text_valid_mask = text_mask.unsqueeze(-1).float()
            # ... (unchanged) ...
            
            # (Keeping existing sync logic but assuming it is safe or disabled)
            text_pooled = (text_features * text_valid_mask).sum(1) / \
                         text_valid_mask.sum(1).clamp(min=1)
            
            pos = (self.sync_head(latent_t, text_pooled, valid_mask) *
                   valid_mask.float()).sum(1) / valid_mask.sum(1).clamp(min=1)
            
            neg = (self.sync_head(latent_t, text_pooled[torch.randperm(B, device=device)], valid_mask) *
                   valid_mask.float()).sum(1) / valid_mask.sum(1).clamp(min=1)
            
            sync_loss = F.relu(0.5 + neg - pos).mean()
            if torch.isnan(sync_loss): sync_loss = torch.tensor(0.0, device=device)
        
        # Loss 4: Flow matching (CRITICAL - let NaN propagate!)
        flow_loss = self.flow_loss_fn(v_flow, v_target, mask=valid_mask)
        
        # Total loss
        total_loss = flow_loss + \
                    self.W_PRIOR * prior_loss + \
                    self.W_SYNC * sync_loss + \
                    self.W_LENGTH * length_loss
        
        v_pred_train = v_flow + lambda_t * v_prior.detach() if v_prior is not None else v_flow
        
        result = {
            'total': total_loss,
            'flow': flow_loss,
            'prior': prior_loss,
            'sync': sync_loss,
            'length': length_loss,
            'predicted_latent': latent_t.detach(),
            'velocity_pred': v_pred_train.detach(),
            'velocity_target': v_target.detach(),
            'v_flow': v_flow.detach(),
            'v_prior': v_prior.detach() if v_prior is not None else None,
            'timestep': t.detach(),
            'valid_mask': valid_mask.detach()
        }
        
        if return_attn_weights:
            result['attn_weights'] = out[1]
        
        return result