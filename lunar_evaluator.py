#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lunar_evaluator.py

This file implements the LunarMoETeacher model – a Mixture of Experts (MoE) teacher model 
for assessing the quality of pixel art generated by our system. The model is designed to work 
with 128×128 pixel art images and consists of the following components:

1. PixelArtFeatureExtractor:
   - Extracts rich features from input images using three specialized branches:
     a. Edge detection (captures outlines and sharp transitions)
     b. Color analysis (extracts global color patterns)
     c. Detail preservation (maintains fine pixel-level details)
   - The outputs of these branches are fused into a common feature representation.

2. PixelArtAttention:
   - Implements a memory‑efficient multi‑head self‑attention mechanism with relative positional encodings.
   - This module refines the features by capturing long‑range dependencies.

3. ExpertBlock:
   - A building block that processes input features through a convolution, self‑attention, and residual connection.
   - It includes a learnable layer scale to help with gradient flow.

4. LunarMoETeacher:
   - Contains:
     • A feature extractor to compute initial image features.
     • A set of expert networks (each built by stacking several ExpertBlocks) that process these features.
     • A gating network that computes dynamic weights for each expert.
     • Quality assessment heads for each expert that produce quality scores.
     • Embedding networks that generate style and prompt embeddings.
   - The final quality score is computed as a weighted sum of the expert quality scores.
   - During inference, the model outputs:
     • quality_scores: A vector of quality metrics (e.g., edge quality, color consistency, detail, overall)
     • expert_weights: The dynamic weights for each expert
     • style_embedding: A style embedding vector derived from combined expert outputs
     • prompt_embedding: A prompt embedding vector (for potential conditioning)
     • feature_maps: Optionally, the intermediate expert outputs (for visualization when not training)

Author: Your Name
Date: YYYY-MM-DD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def mish(x):
    """Mish activation function."""
    return x * torch.tanh(F.softplus(x))


# -------------------------------
# Feature Extraction Module
# -------------------------------

class PixelArtFeatureExtractor(nn.Module):
    """
    Specialized feature extractor for pixel art.
    
    This module processes input images through three branches:
      - Edge branch: Captures outlines using depthwise convolutions.
      - Color branch: Analyzes color patterns with larger kernels.
      - Detail branch: Preserves fine details.
      
    The outputs are concatenated and fused into a common feature representation.
    """
    def __init__(self, in_channels=3, dropout_rate=0.1, feature_dim=128):
        super(PixelArtFeatureExtractor, self).__init__()
        # Initial convolution to process input
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )
        # Edge detection branch
        self.edge_branch = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )
        # Color analysis branch
        self.color_branch = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )
        # Detail preservation branch
        self.detail_branch = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )
        self.dropout = nn.Dropout(dropout_rate)
        # Fuse the features from all branches
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, feature_dim, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(feature_dim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        edge_features = self.edge_branch(x)
        color_features = self.color_branch(x)
        detail_features = self.detail_branch(x)
        combined = torch.cat([edge_features, color_features, detail_features], dim=1)
        combined = self.dropout(combined)
        return self.fusion(combined)


# -------------------------------
# Self-Attention Module
# -------------------------------

class PixelArtAttention(nn.Module):
    """
    Memory-efficient multi-head self-attention module for pixel art.
    
    This module processes input features in chunks to reduce memory usage.
    It also uses relative positional encoding to capture spatial relationships.
    """
    def __init__(self, in_channels, num_heads=8, rel_pos_size=8, dropout=0.1, chunk_size=64):
        super(PixelArtAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.chunk_size = chunk_size
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.rel_pos_h = nn.Parameter(torch.randn(1, num_heads, rel_pos_size, 1) * 0.02)
        self.rel_pos_w = nn.Parameter(torch.randn(1, num_heads, 1, rel_pos_size) * 0.02)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
        # Caching for relative positional encoding
        self.register_buffer('rel_pos_cache', None)
        self.register_buffer('last_spatial_shapes', torch.zeros(2))
        
    def _process_qkv_efficiently(self, x, chunk_size=None):
        # Reduce memory usage by processing in smaller chunks
        chunk_size = chunk_size or min(self.chunk_size, 32)  # Reduce default chunk size
        B, C, H, W = x.shape
        N = H * W
        
        # Process QKV more efficiently
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H, W)
        qkv = qkv.permute(0, 1, 2, 4, 5, 3)
        qkv = qkv.reshape(B, 3, self.num_heads, N, self.head_dim)
        
        # Free memory
        del x
        torch.cuda.empty_cache()
        
        chunks = []
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk = qkv[..., i:end, :].contiguous()  # Make contiguous for better memory layout
            chunks.append(chunk)
        
        # Free memory
        del qkv
        torch.cuda.empty_cache()
        
        return chunks, H, W, N

    def _process_rel_pos(self, H, W, device):
        current_spatial = torch.tensor([H, W], device=device)
        if self.rel_pos_cache is not None and torch.all(current_spatial == self.last_spatial_shapes):
            return self.rel_pos_cache
        rel_h = F.interpolate(self.rel_pos_h, size=(H, 1), mode='bilinear', align_corners=True).to(device)
        rel_w = F.interpolate(self.rel_pos_w, size=(1, W), mode='bilinear', align_corners=True).to(device)
        rel_h = rel_h.expand(-1, -1, -1, W)
        rel_w = rel_w.expand(-1, -1, H, -1)
        rel_pos = (rel_h + rel_w).reshape(1, self.num_heads, H * W)
        rel_pos = rel_pos.unsqueeze(-1)
        self.rel_pos_cache = rel_pos
        self.last_spatial_shapes = current_spatial
        return rel_pos

    @torch.amp.autocast('cuda')
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Process in chunks with gradient checkpointing if training
        if self.training:
            qkv_chunks, H, W, N = torch.utils.checkpoint.checkpoint(self._process_qkv_efficiently, x)
        else:
            qkv_chunks, H, W, N = self._process_qkv_efficiently(x)
        
        out = torch.zeros(B, self.num_heads, N, self.head_dim, device=x.device, dtype=x.dtype)
        rel_pos = self._process_rel_pos(H, W, x.device)
        
        # Process chunks with explicit memory management
        for i, qkv in enumerate(qkv_chunks):
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            
            # Compute attention scores in smaller chunks
            attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            chunk_size = qkv.size(-2)
            chunk_rel_pos = rel_pos[..., i:i+chunk_size, :]
            attn = attn + chunk_rel_pos
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            # Compute output for this chunk
            chunk_out = torch.matmul(attn, v)
            out[..., i:i+chunk_size, :] = chunk_out
            
            # Free memory
            del attn, chunk_out
            torch.cuda.empty_cache()
        
        # Reshape and project output
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


# -------------------------------
# Expert Block and Mixture of Experts (MoE) Teacher Model
# -------------------------------

class ExpertBlock(nn.Module):
    """
    ExpertBlock: Processes features using convolutional layers, self-attention, 
    and a residual connection with layer scaling for improved gradient flow.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, rel_pos_size=8, layer_scale_init=0.1):
        super(ExpertBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate)
        )
        self.attention = PixelArtAttention(out_channels, rel_pos_size=rel_pos_size, dropout=dropout_rate)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * layer_scale_init)
    
    def _forward_path(self, x):
        out = self.conv1(x)
        out = self.attention(out)
        out = self.conv2(out)
        return out * self.layer_scale
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        if self.training:
            # Use gradient checkpointing for the main computation path
            out = torch.utils.checkpoint.checkpoint(self._forward_path, x)
        else:
            out = self._forward_path(x)
        
        return F.leaky_relu(out + identity, 0.2)


class LunarMoETeacher(nn.Module):
    """
    LunarMoETeacher: A Mixture of Experts (MoE) teacher model for assessing pixel art quality.
    
    Components:
      - Feature extractor: Processes input images into a compact representation.
      - Experts: A set of expert networks (built from multiple ExpertBlocks) that specialize in quality assessment.
      - Gating network: Dynamically weights the expert outputs.
      - Quality heads: Evaluate the quality of the output from each expert.
      - Embedding networks: Generate style and prompt embeddings for further conditioning.
      
    The final output includes quality scores (normalized via sigmoid), expert weights, and embeddings.
    """
    def __init__(
        self,
        num_experts=4,
        feature_dim=128,
        dropout_rate=0.1,
        rel_pos_size=8,
        use_checkpointing=True,
        expert_layers=3,
        intermediate_dim=256,
        embedding_dim=64
    ):
        super(LunarMoETeacher, self).__init__()
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.rel_pos_size = rel_pos_size
        self.use_checkpointing = use_checkpointing
        self.expert_layers = expert_layers
        self.intermediate_dim = intermediate_dim
        self.embedding_dim = embedding_dim
        
        # Feature extractor
        self.feature_extractor = PixelArtFeatureExtractor(in_channels=3, dropout_rate=dropout_rate, feature_dim=128)
        
        # Build expert networks
        self.experts = nn.ModuleList([
            self._build_expert_network() for _ in range(num_experts)
        ])
        
        # Gating network for expert weighting
        self.gate = self._build_gating_network()
        
        # Quality assessment heads
        self.quality_heads = nn.ModuleList([
            self._build_quality_head() for _ in range(num_experts)
        ])
        
        # Semantic matching head for prompt-image correspondence
        self.semantic_head = self._build_semantic_head()
        
        # Embedding networks for style and prompt representations
        self.style_net = self._build_embedding_network()
        self.prompt_net = self._build_embedding_network()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_expert_network(self):
        layers = []
        in_dim = 128  # Input feature dimension from feature extractor
        for _ in range(self.expert_layers):
            layers.append(
                ExpertBlock(
                    in_channels=in_dim,
                    out_channels=self.feature_dim,
                    dropout_rate=self.dropout_rate,
                    rel_pos_size=self.rel_pos_size
                )
            )
            in_dim = self.feature_dim
        return nn.Sequential(*layers)
    
    def _build_gating_network(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.intermediate_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.intermediate_dim, self.num_experts),
            nn.Softmax(dim=1)
        )
    
    def _build_quality_head(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.intermediate_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.intermediate_dim // 4, 4)  # Four quality metrics
        )
    
    def _build_semantic_head(self):
        """Build semantic matching head to evaluate prompt-image correspondence."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.intermediate_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.intermediate_dim // 2, 1),  # Single semantic matching score
            nn.Sigmoid()  # Normalize to [0,1]
        )
    
    def _build_embedding_network(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.intermediate_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.intermediate_dim // 2, self.embedding_dim)
        )
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    @torch.amp.autocast('cuda')
    def forward(self, x, prompt_embedding=None):
        # Extract features from the input image
        if self.use_checkpointing and self.training:
            features = torch.utils.checkpoint.checkpoint(self.feature_extractor, x)
        else:
            features = self.feature_extractor(x)
        
        # Compute expert weights using the gating network
        expert_weights = self.gate(features)  # Shape: [B, num_experts]
        
        # Process the features through each expert and obtain quality scores
        expert_outputs = []
        quality_scores = []
        for expert, quality_head in zip(self.experts, self.quality_heads):
            expert_feat = expert(features)
            expert_outputs.append(expert_feat)
            quality_score = quality_head(expert_feat)
            quality_scores.append(quality_score)
            # Optionally clear intermediate outputs to free memory if needed
            torch.cuda.empty_cache()
        
        # Stack and combine quality scores from all experts
        quality_tensor = torch.stack(quality_scores, dim=1)  # [B, num_experts, 4]
        weighted_quality = torch.sum(quality_tensor * expert_weights.unsqueeze(-1), dim=1)  # [B, 4]
        
        # Generate style and prompt embeddings from the weighted expert features
        combined_features = torch.stack([feat.mean(dim=[2, 3]) for feat in expert_outputs], dim=1)
        combined_features = torch.sum(combined_features * expert_weights.unsqueeze(-1), dim=1)  # [B, feature_dim]
        style_embedding = self.style_net(combined_features.unsqueeze(-1).unsqueeze(-1))
        prompt_embedding = self.prompt_net(combined_features.unsqueeze(-1).unsqueeze(-1))
        
        # Compute semantic matching score if prompt embedding is provided
        semantic_score = None
        if prompt_embedding is not None:
            # Use the first expert's features for semantic matching
            semantic_features = expert_outputs[0]
            semantic_score = self.semantic_head(semantic_features)
            
            # Adjust score based on prompt similarity
            prompt_similarity = F.cosine_similarity(prompt_embedding, prompt_embedding.detach(), dim=1)
            semantic_score = semantic_score * prompt_similarity.unsqueeze(1)
        
        # Clear temporary tensors to save memory
        del quality_tensor, combined_features
        torch.cuda.empty_cache()
        
        return {
            'quality_scores': torch.sigmoid(weighted_quality),
            'expert_weights': expert_weights,
            'style_embedding': style_embedding,
            'prompt_embedding': prompt_embedding,
            'semantic_score': semantic_score,
            'feature_maps': expert_outputs if not self.training else None
        }

# If this module is run directly, print a summary of the model
if __name__ == "__main__":
    model = LunarMoETeacher()
    print(model)
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 128, 128).cuda()
    output = model(dummy_input)
    print("Quality scores shape:", output['quality_scores'].shape)
    print("Expert weights shape:", output['expert_weights'].shape)
    print("Style embedding shape:", output['style_embedding'].shape)
    print("Prompt embedding shape:", output['prompt_embedding'].shape)
