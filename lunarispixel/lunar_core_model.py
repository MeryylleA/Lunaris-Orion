import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """
    Residual block with batch normalization and dropout.
    """
    def __init__(self, channels, dropout_rate=0.1, activation=nn.LeakyReLU(0.2)):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out

class SelfAttention2d(nn.Module):
    """
    Self-attention module for 2D feature maps.
    """
    def __init__(self, in_channels):
        super(SelfAttention2d, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        query = self.query_conv(x).view(B, -1, W*H)
        key = self.key_conv(x).view(B, -1, W*H)
        value = self.value_conv(x).view(B, -1, W*H)

        attn = torch.bmm(query.permute(0, 2, 1), key)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(value, attn.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        return out

class LunarCoreEncoder(nn.Module):
    """
    Encoder network for the LunarCore VAE updated for 128x128 images.
    Downsamples input through five blocks:
      - Initial conv: 128x128 -> 128x128 with 64 channels
      - Down1: 128x128 -> 64x64, channels: 64 -> 128 (with self-attention)
      - Down2: 64x64 -> 32x32, channels: 128 -> 256 (with self-attention)
      - Down3: 32x32 -> 16x16, channels: 256 -> 512
      - Down4: 16x16 -> 8x8, channels: 512 -> 512
    It then flattens and projects to latent mean and log-variance.
    """
    def __init__(self, latent_dim=128, activation=nn.LeakyReLU(0.2)):
        super(LunarCoreEncoder, self).__init__()
        self.activation = activation

        # Initial convolution: from 3 channels to 64, resolution remains 128x128
        self.conv_initial = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # initial processing
        self.bn_initial = nn.BatchNorm2d(64)
        self.res_initial = ResNetBlock(64, activation=activation)  # preserves details

        # First downsampling block: 128x128 -> 64x64, channels: 64 -> 128
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.res1 = ResNetBlock(128, activation=activation)
        self.attn1 = SelfAttention2d(128)  # self-attention to capture global patterns

        # Second downsampling block: 64x64 -> 32x32, channels: 128 -> 256
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.res2 = ResNetBlock(256, activation=activation)
        self.attn2 = SelfAttention2d(256)

        # Third downsampling block: 32x32 -> 16x16, channels: 256 -> 512
        self.down3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.res3 = ResNetBlock(512, activation=activation)

        # Fourth downsampling block: 16x16 -> 8x8, channels: 512 -> 512 (keeping same channel count)
        self.down4 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.res4 = ResNetBlock(512, activation=activation)

        # Updated flattened tensor size: 512 channels * 8 * 8
        self.flatten_size = 512 * 8 * 8

        # Fully connected layers for latent projection
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # x: input of shape (B, 3, 128, 128)
        
        # Initial processing at 128x128
        x0 = self.conv_initial(x)            # (B, 64, 128, 128)
        x0 = self.bn_initial(x0)
        x0 = self.activation(x0)
        x0 = self.res_initial(x0)            # Save as skip connection for later

        # First downsampling: 128 -> 64
        x1 = self.down1(x0)                  # (B, 128, 64, 64)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        x1 = self.res1(x1)
        x1 = self.attn1(x1)                  # Save skip connection at 64x64

        # Second downsampling: 64 -> 32
        x2 = self.down2(x1)                  # (B, 256, 32, 32)
        x2 = self.bn2(x2)
        x2 = self.activation(x2)
        x2 = self.res2(x2)
        x2 = self.attn2(x2)                  # Save skip connection at 32x32

        # Third downsampling: 32 -> 16
        x3 = self.down3(x2)                  # (B, 512, 16, 16)
        x3 = self.bn3(x3)
        x3 = self.activation(x3)
        x3 = self.res3(x3)                  # Save skip connection at 16x16

        # Fourth downsampling: 16 -> 8
        x4 = self.down4(x3)                  # (B, 512, 8, 8)
        x4 = self.bn4(x4)
        x4 = self.activation(x4)
        x4 = self.res4(x4)                  # Final feature map

        # Flatten and project to latent space
        flat = self.flatten(x4)              # (B, 512*8*8)
        mean = self.fc_mean(flat)
        logvar = self.fc_logvar(flat)

        # Return latent projections and skip connections for use in decoder
        return mean, logvar, (x0, x1, x2, x3)

class LunarCoreDecoder(nn.Module):
    """
    Decoder network for the LunarCore VAE.
    Reconstructs 16x16 pixel art images from latent vectors through
    transposed convolutions, residual blocks, and self-attention mechanisms.
    """
    def __init__(self, latent_dim=128, activation=nn.LeakyReLU(0.2)):
        super(LunarCoreDecoder, self).__init__()
        self.activation = activation

        # Project latent vector to feature map of shape (512, 8, 8)
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))

        # Upsampling Block 1: 8x8 -> 16x16
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.res_up1 = ResNetBlock(1024, activation=activation)  # after concatenating skip from encoder (x3: 512 channels)
        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1)           # reduce channels back to 512

        # Upsampling Block 2: 16x16 -> 32x32
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.res_up2 = ResNetBlock(512, activation=activation)         # concat with skip from encoder (x2: 256 channels) -> 256+256=512
        self.reduce2 = nn.Conv2d(512, 256, kernel_size=1)              # reduce channels

        # Upsampling Block 3: 32x32 -> 64x64
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.res_up3 = ResNetBlock(256, activation=activation)         # concat with skip from encoder (x1: 128 channels) -> 128+128=256
        self.reduce3 = nn.Conv2d(256, 128, kernel_size=1)

        # Upsampling Block 4: 64x64 -> 128x128
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.res_up4 = ResNetBlock(128, activation=activation)         # concat with skip from encoder (x0: 64 channels) -> 64+64=128
        self.reduce4 = nn.Conv2d(128, 64, kernel_size=1)

        # Final output layer: map to 3 channels (RGB)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, z, skips):
        # Unpack skip connections: x0 (128x128), x1 (64x64), x2 (32x32), x3 (16x16)
        x0, x1, x2, x3 = skips

        # Initial projection from latent space to (512, 8, 8)
        x = self.fc(z)
        x = self.activation(x)
        x = self.unflatten(x)  # (B, 512, 8, 8)

        # Upsampling Block 1: 8x8 -> 16x16
        x = self.up1(x)       # (B, 512, 16, 16)
        x = self.bn1(x)
        x = self.activation(x)
        # Concatenate with skip from encoder x3 (16x16, 512 channels)
        x = torch.cat([x, x3], dim=1)  # (B, 1024, 16, 16)
        x = self.res_up1(x)            # merge features
        x = self.reduce1(x)            # reduce channels from 1024 to 512

        # Upsampling Block 2: 16x16 -> 32x32
        x = self.up2(x)       # (B, 256, 32, 32)
        x = self.bn2(x)
        x = self.activation(x)
        # Concatenate with skip from encoder x2 (32x32, 256 channels)
        x = torch.cat([x, x2], dim=1)  # (B, 512, 32, 32)
        x = self.res_up2(x)
        x = self.reduce2(x)            # reduce to 256 channels

        # Upsampling Block 3: 32x32 -> 64x64
        x = self.up3(x)       # (B, 128, 64, 64)
        x = self.bn3(x)
        x = self.activation(x)
        # Concatenate with skip from encoder x1 (64x64, 128 channels)
        x = torch.cat([x, x1], dim=1)  # (B, 256, 64, 64)
        x = self.res_up3(x)
        x = self.reduce3(x)            # reduce to 128 channels

        # Upsampling Block 4: 64x64 -> 128x128
        x = self.up4(x)       # (B, 64, 128, 128)
        x = self.bn4(x)
        x = self.activation(x)
        # Concatenate with skip from encoder x0 (128x128, 64 channels)
        x = torch.cat([x, x0], dim=1)  # (B, 128, 128, 128)
        x = self.res_up4(x)
        x = self.reduce4(x)            # reduce to 64 channels

        # Final output layer
        x = self.conv_out(x)           # (B, 3, 128, 128)
        x = torch.tanh(x)              # Normalize output to [-1, 1]

        return x

class LunarCoreVAE(nn.Module):
    """
    LunarCore Variational Autoencoder for pixel art generation.
    Combines encoder and decoder networks with a VAE architecture.
    """
    def __init__(self, latent_dim=128, activation=nn.LeakyReLU(0.2)):
        super(LunarCoreVAE, self).__init__()
        self.encoder = LunarCoreEncoder(latent_dim, activation)
        self.decoder = LunarCoreDecoder(latent_dim, activation)

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick for VAE training.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar, skips = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z, skips)
        return recon_x, mean, logvar
