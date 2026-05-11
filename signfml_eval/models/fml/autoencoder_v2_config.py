"""
Configuration for Stage 1 Autoencoder v2
Easy to tune parameters to reach target total params

Targets:
- Total: 78M
- Encoder: 30M (38%)
- Decoder: 48M (62%)
"""

# =============================================================================
# CONFIG: OPTIMAL (78M target)
# =============================================================================

CONFIG_78M = {
    # Input/Output
    "pose_dim": 214,
    "latent_dim": 256,
    "hidden_dim": 512,
    
    # Encoder
    "encoder_layers": 6,
    "local_heads": 2,
    "medium_heads": 2,
    "global_heads": 4,
    "encoder_ffn_hidden": 1024,  # Reduced for 78M
    
    # Decoder
    "decoder_coarse_layers": 3,
    "decoder_medium_layers": 3,
    "decoder_fine_layers": 4,
    "num_heads": 8,
    "decoder_ffn_hidden": 1024,  # Reduced for 78M
    
    # Regularization
    "dropout": 0.1
}

# =============================================================================
# CONFIG: LARGE (100M for comparison)
# =============================================================================

CONFIG_100M = {
    "pose_dim": 214,
    "latent_dim": 256,
    "hidden_dim": 512,
    
    "encoder_layers": 6,
    "local_heads": 2,
    "medium_heads": 2,
    "global_heads": 4,
    "encoder_ffn_hidden": 1536,
    
    "decoder_coarse_layers": 4,
    "decoder_medium_layers": 4,
    "decoder_fine_layers": 6,
    "num_heads": 8,
    "decoder_ffn_hidden": 1536,
    
    "dropout": 0.1
}

# =============================================================================
# CONFIG: SMALL (60M for testing)
# =============================================================================

CONFIG_60M = {
    "pose_dim": 214,
    "latent_dim": 256,
    "hidden_dim": 512,
    
    "encoder_layers": 6,
    "local_heads": 2,
    "medium_heads": 2,
    "global_heads": 4,
    "encoder_ffn_hidden": 768,
    
    "decoder_coarse_layers": 2,
    "decoder_medium_layers": 2,
    "decoder_fine_layers": 3,
    "num_heads": 8,
    "decoder_ffn_hidden": 768,
    
    "dropout": 0.1
}

# Default config
DEFAULT_CONFIG = CONFIG_78M


def get_config(name="78M"):
    """Get config by name"""
    configs = {
        "78M": CONFIG_78M,
        "100M": CONFIG_100M,
        "60M": CONFIG_60M,
    }
    return configs.get(name, CONFIG_78M)


if __name__ == "__main__":
    import torch
    from autoencoder_v2 import UnifiedPoseAutoencoder
    
    for name, config in [("60M", CONFIG_60M), ("78M", CONFIG_78M), ("100M", CONFIG_100M)]:
        print(f"\n{'='*50}")
        print(f"Testing CONFIG_{name}")
        print(f"{'='*50}")
        
        model = UnifiedPoseAutoencoder(
            pose_dim=config["pose_dim"],
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            encoder_layers=config["encoder_layers"],
            decoder_coarse_layers=config["decoder_coarse_layers"],
            decoder_medium_layers=config["decoder_medium_layers"],
            decoder_fine_layers=config["decoder_fine_layers"],
            num_heads=config["num_heads"],
            local_heads=config["local_heads"],
            medium_heads=config["medium_heads"],
            global_heads=config["global_heads"],
            ffn_hidden=config.get("encoder_ffn_hidden", config.get("decoder_ffn_hidden", 1024)),
            dropout=config["dropout"]
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        
        print(f"Encoder: {encoder_params/1e6:.1f}M ({encoder_params/total_params*100:.1f}%)")
        print(f"Decoder: {decoder_params/1e6:.1f}M ({decoder_params/total_params*100:.1f}%)")
        print(f"Total: {total_params/1e6:.1f}M")
