"""
Stage 1: Pose VQ-VAE (Discrete Latents)
Integrates existing PoseEncoder/PoseDecoder with Vector Quantization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .autoencoder import PoseEncoder, PoseDecoder

class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantizer with EMA updates.
    Ref: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon
        
        # Initialize embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
    def forward(self, inputs):
        """
        inputs: [B, T, D]
        """
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
            
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return {
            'quantized': quantized,
            'loss': commitment_loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices.view(input_shape[:-1])
        }

class PoseVQVAE(nn.Module):
    """
    VQ-VAE for 3D Poses.
    Structure: Encoder -> VQ -> Decoder
    """
    def __init__(
        self,
        pose_dim=534,          # 178 keypoints * 3
        latent_dim=256,
        hidden_dim=512,
        num_embeddings=1024,   # Codebook size
        encoder_layers=4,
        decoder_layers=6,      # Hierarchical
        num_heads=8,
        dropout=0.1,
        commitment_cost=0.25
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
        
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost
        )
        
        self.decoder = PoseDecoder(
            latent_dim=latent_dim,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_coarse_layers=decoder_layers // 2,
            num_medium_layers=decoder_layers // 2,
            num_fine_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, pose, mask=None):
        """
        Args:
            pose: [B, T, pose_dim]
            mask: [B, T]
        """
        # 1. Encode -> [B, T, latent_dim]
        z_e = self.encoder(pose, mask)
        
        # 2. Quantize
        vq_out = self.vq(z_e)
        z_q = vq_out['quantized']
        
        # 3. Decode
        reconstruction = self.decoder(z_q, mask)
        
        return {
            'reconstruction': reconstruction,
            'z_e': z_e,
            'z_q': z_q,
            'vq_loss': vq_out['loss'],
            'perplexity': vq_out['perplexity'],
            'indices': vq_out['encoding_indices']
        }
    
    def encode_indices(self, pose, mask=None):
        """Get codebook indices for Stage 2 training"""
        z_e = self.encoder(pose, mask)
        vq_out = self.vq(z_e)
        return vq_out['encoding_indices']

if __name__ == "__main__":
    # Quick Test
    model = PoseVQVAE()
    print(f"PoseVQVAE Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    x = torch.randn(2, 50, 534)
    out = model(x)
    print("Output shape:", out['reconstruction'].shape)
    print("VQ Loss:", out['vq_loss'].item())
