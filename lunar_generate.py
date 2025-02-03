#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lunar_generate.py

LunarisCoreVAE: A variational autoencoder (VAE) for generating pixel art.
The model uses an encoder to compress 128×128 images into a latent vector
and a decoder to reconstruct images from the latent space. Skip connections
and residual blocks are used to preserve fine details. The final layer uses
a tanh activation to normalize the output to the range [-1, 1].

Developer: Moon Cloud Services
Date: 02/02/25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Utility Functions and Modules
# -------------------------------

def mish(x):
    """Mish activation function."""
    return x * torch.tanh(F.softplus(x))

class ResBlock(nn.Module):
    """
    Residual Block with two convolutional layers, GroupNorm, and Mish activation.
    It also includes a skip connection to preserve details.
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.Mish()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.Mish()
        )
        # Shortcut: if dimensions differ, use a 1x1 conv to match channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return mish(out + identity)

# Optional self-attention module (if you decide to add it later)
class SelfAttention2d(nn.Module):
    """
    A simple self-attention module for 2D feature maps.
    This implementation computes the attention over the spatial dimensions.
    """
    def __init__(self, in_channels):
        super(SelfAttention2d, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W)          # [B, C//8, N]
        proj_key = self.key_conv(x).view(B, -1, H * W)                # [B, C//8, N]
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)       # [B, N, N]
        attention = F.softmax(energy, dim=-1)                         # [B, N, N]
        proj_value = self.value_conv(x).view(B, -1, H * W)            # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))         # [B, C, N]
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

# -------------------------------
# Model Components: Encoder and Decoder
# -------------------------------

class Encoder(nn.Module):
    """
    Encoder network that compresses 128x128 pixel art images to a latent vector.
    It uses 4 downsampling blocks to reduce the spatial dimensions:
      128 → 64 → 32 → 16 → 8.
    Each block includes a convolutional layer, GroupNorm, Mish activation, and a ResBlock.
    """
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        # Down Block 1: 128x128 -> 64x64, channels: 3 -> 64
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            ResBlock(64, 64)
        )
        # Down Block 2: 64x64 -> 32x32, channels: 64 -> 128
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            ResBlock(128, 128)
        )
        # Down Block 3: 32x32 -> 16x16, channels: 128 -> 256
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.Mish(),
            ResBlock(256, 256)
        )
        # Down Block 4: 16x16 -> 8x8, channels: 256 -> 512
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.Mish(),
            ResBlock(512, 512)
        )

        # Fully connected layers for latent space projection
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input image tensor of shape [B, 3, 128, 128]

        Returns:
            mu (Tensor): Mean vector of latent distribution
            logvar (Tensor): Log-variance vector of latent distribution
            skips (list): List of intermediate feature maps for skip connections
        """
        skips = []
        x = self.down1(x)    # [B, 64, 64, 64]
        skips.append(x)
        x = self.down2(x)    # [B, 128, 32, 32]
        skips.append(x)
        x = self.down3(x)    # [B, 256, 16, 16]
        skips.append(x)
        x = self.down4(x)    # [B, 512, 8, 8]
        # No skip saved for the last layer (or optionally add it)

        # Flatten and project to latent space
        x_flat = self.flatten(x)  # 512 * 8 * 8 = 32768
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar, skips

class Decoder(nn.Module):
    """
    Decoder network that reconstructs 128x128 images from the latent vector.
    It first projects the latent vector to a tensor of shape [B, 512, 8, 8] and then
    applies a sequence of upsampling blocks. Skip connections from the encoder are added
    to refine the reconstruction.
    """
    def __init__(self, latent_dim=256):
        super(Decoder, self).__init__()
        # Project latent vector to [B, 512, 8, 8]
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        # Upsampling Block 1: 8x8 -> 16x16, combine with skip3 (256 channels)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.Mish()
        )
        # Upsampling Block 2: 16x16 -> 32x32, combine with skip2 (128 channels)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish()
        )
        # Upsampling Block 3: 32x32 -> 64x64, combine with skip1 (64 channels)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish()
        )
        # Upsampling Block 4: 64x64 -> 128x128, no skip connection here
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.Mish()
        )
        # Final output layer: map to 3 channels (RGB) with tanh activation
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, z, skips):
        """
        Forward pass through the decoder.

        Args:
            z (Tensor): Latent vector of shape [B, latent_dim]
            skips (list): List of skip connection feature maps from the encoder

        Returns:
            Tensor: Reconstructed image of shape [B, 3, 128, 128]
        """
        B = z.size(0)
        # Project and reshape latent vector to [B, 512, 8, 8]
        x = self.fc(z)
        x = x.view(B, 512, 8, 8)

        # Upsample with skip connections (using addition)
        x = self.up1(x)  # [B, 256, 16, 16]
        if len(skips) >= 3:
            # Ensure the skip from down block 3 has same shape as x
            x = x + skips[2]

        x = self.up2(x)  # [B, 128, 32, 32]
        if len(skips) >= 2:
            x = x + skips[1]

        x = self.up3(x)  # [B, 64, 64, 64]
        if len(skips) >= 1:
            x = x + skips[0]

        x = self.up4(x)  # [B, 32, 128, 128]

        # Final output layer
        x = self.final_conv(x)
        x = torch.tanh(x)  # Normalize to [-1, 1]
        return x

class LunarisCoreVAE(nn.Module):
    """
    LunarisCoreVAE: Variational Autoencoder for Pixel Art Generation.

    This model consists of:
      - Encoder: Compresses input images to a latent space.
      - Reparameterization: Samples from the latent space.
      - Decoder: Reconstructs images from latent vectors.

    The model is designed to work with 128x128 pixel art images.
    """
    def __init__(self, latent_dim=256):
        super(LunarisCoreVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * sigma.

        Args:
            mu (Tensor): Mean tensor of latent distribution.
            logvar (Tensor): Log variance tensor of latent distribution.

        Returns:
            Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (Tensor): Input image tensor of shape [B, 3, 128, 128]

        Returns:
            tuple: (reconstructed_image, mu, logvar)
        """
        mu, logvar, skips = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z, skips)
        return reconstruction, mu, logvar

    def sample(self, num_samples):
        """
        Generate new images by sampling random latent vectors.

        Args:
            num_samples (int): Number of images to generate.

        Returns:
            Tensor: Generated images of shape [num_samples, 3, 128, 128]
        """
        z = torch.randn(num_samples, self.latent_dim, device=next(self.parameters()).device)
        # No skip connections when generating from pure noise
        reconstruction = self.decoder(z, skips=[])
        return reconstruction

# If this script is run directly, print a summary of the model.
if __name__ == "__main__":
    model = LunarisCoreVAE(latent_dim=256)
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 128, 128)
    reconstruction, mu, logvar = model(dummy_input)
    print("Reconstruction shape:", reconstruction.shape)
    print("Latent mean shape:", mu.shape)
    print("Latent logvar shape:", logvar.shape)
