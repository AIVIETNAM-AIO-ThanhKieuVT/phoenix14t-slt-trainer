#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMA (Exponential Moving Average) for Model Parameters
======================================================
Author: SignFML Research
Date: 21 Jan 2026

Critical for diffusion/flow models - maintains a smoothed copy of model weights.
Uses the EMA model for inference, which typically produces better results.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Usage:
        ema = EMA(model, decay=0.9999)
        
        # After each training step:
        ema.update()
        
        # For inference:
        with ema.average_parameters():
            model(x)  # Uses EMA weights
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        min_decay: float = 0.0
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (0.9999 is common, higher = slower update)
            warmup_steps: Number of steps before reaching target decay
            min_decay: Starting decay value during warmup
        """
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.min_decay = min_decay
        self.step_count = 0
        
        # Create shadow copy of parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def get_decay(self) -> float:
        """Get current decay value (with warmup)."""
        if self.warmup_steps > 0:
            # Linear warmup of decay
            progress = min(self.step_count / self.warmup_steps, 1.0)
            return self.min_decay + (self.decay - self.min_decay) * progress
        return self.decay
    
    @torch.no_grad()
    def update(self):
        """Update shadow parameters with current model parameters."""
        decay = self.get_decay()
        self.step_count += 1
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA update: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for inference)."""
        self.backup.clear()
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters after inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
    
    class _ContextManager:
        """Context manager for using EMA weights."""
        def __init__(self, ema):
            self.ema = ema
        
        def __enter__(self):
            self.ema.apply_shadow()
            return self.ema.model
        
        def __exit__(self, *args):
            self.ema.restore()
    
    def average_parameters(self):
        """Context manager to use EMA weights temporarily."""
        return self._ContextManager(self)
    
    def state_dict(self):
        """Get state dict for saving."""
        return {
            'shadow': self.shadow,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.shadow = state_dict['shadow']
        self.step_count = state_dict['step_count']


class EMAModel(nn.Module):
    """
    Wrapper that creates a separate EMA copy of the model.
    Alternative to the above approach - maintains a full model copy.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        
        # Create EMA copy
        self.ema_model = deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA model with current model parameters."""
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def forward(self, *args, **kwargs):
        """Forward pass using EMA model."""
        return self.ema_model(*args, **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing EMA...")
    
    # Create dummy model
    model = nn.Linear(10, 10)
    ema = EMA(model, decay=0.99, warmup_steps=100)
    
    # Simulate training
    for step in range(200):
        # Simulate parameter update
        with torch.no_grad():
            model.weight.data += torch.randn_like(model.weight) * 0.01
        
        # Update EMA
        ema.update()
        
        if step % 50 == 0:
            print(f"Step {step}: decay={ema.get_decay():.4f}")
    
    # Test context manager
    original_weight = model.weight.data.clone()
    
    with ema.average_parameters():
        ema_weight = model.weight.data.clone()
        print(f"\nEMA weight differs from current: {not torch.allclose(original_weight, ema_weight)}")
    
    # Verify restoration
    restored = torch.allclose(model.weight.data, original_weight)
    print(f"Weights restored after context: {restored}")
    
    print("\n✅ EMA test passed!")
