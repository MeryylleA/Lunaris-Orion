import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
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
    def __init__(self, latent_dim=128, activation=nn.LeakyReLU(0.2)):
        super(LunarCoreEncoder, self).__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResNetBlock(64, activation=activation)

        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.res2 = ResNetBlock(128, activation=activation)
        self.attn1 = SelfAttention2d(128)

        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.res3 = ResNetBlock(256, activation=activation)
        self.attn2 = SelfAttention2d(256)

        self.down3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.res4 = ResNetBlock(512, activation=activation)

        # Calcular o tamanho do tensor achatado
        self.flatten_size = 512 * 2 * 2  # Para imagem 16x16, após 3 downsamplings: 16->8->4->2

        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # Verificar dimensões de entrada
        batch_size = x.size(0)
        
        x1 = self.conv1(x)  # 16x16
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        x1 = self.res1(x1)

        x2 = self.down1(x1)  # 8x8
        x2 = self.bn2(x2)
        x2 = self.activation(x2)
        x2 = self.res2(x2)
        x2 = self.attn1(x2)

        x3 = self.down2(x2)  # 4x4
        x3 = self.bn3(x3)
        x3 = self.activation(x3)
        x3 = self.res3(x3)
        x3 = self.attn2(x3)

        x4 = self.down3(x3)  # 2x2
        x4 = self.bn4(x4)
        x4 = self.activation(x4)
        x4 = self.res4(x4)

        flat = self.flatten(x4)
        mean = self.fc_mean(flat)
        logvar = self.fc_logvar(flat)

        return mean, logvar, (x1, x2, x3)

class LunarCoreDecoder(nn.Module):
    def __init__(self, latent_dim=128, activation=nn.LeakyReLU(0.2)):
        super(LunarCoreDecoder, self).__init__()
        self.activation = activation

        # Calcular o tamanho do tensor inicial
        self.initial_size = 512 * 2 * 2  # Mesmo tamanho do encoder após flatten

        self.fc = nn.Linear(latent_dim, self.initial_size)
        self.unflatten = nn.Unflatten(1, (512, 2, 2))  # Reshape para (batch_size, 512, 2, 2)

        self.res4 = ResNetBlock(512, activation=activation)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 2x2 -> 4x4
        self.bn1 = nn.BatchNorm2d(256)
        self.attn2 = SelfAttention2d(256)
        self.res3 = ResNetBlock(512, activation=activation, dropout_rate=0.2)  # 512 por causa da concatenação

        self.up2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.bn2 = nn.BatchNorm2d(128)
        self.attn1 = SelfAttention2d(128)
        self.res2 = ResNetBlock(256, activation=activation, dropout_rate=0.2)  # 256 por causa da concatenação

        self.up3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.bn3 = nn.BatchNorm2d(64)
        self.res1 = ResNetBlock(128, activation=activation, dropout_rate=0.2)  # 128 por causa da concatenação

        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, z, skips):
        x1, x2, x3 = skips

        # Projetar o vetor latente e remodelar
        x = self.fc(z)
        x = self.activation(x)
        x = self.unflatten(x)  # (batch_size, 512, 2, 2)
        x = self.res4(x)

        # Primeiro upsampling: 2x2 -> 4x4
        x = self.up1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.attn2(x)
        x = torch.cat([x, x3], dim=1)  # Concatenar com skip connection
        x = self.res3(x)

        # Segundo upsampling: 4x4 -> 8x8
        x = self.up2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.attn1(x)
        x = torch.cat([x, x2], dim=1)  # Concatenar com skip connection
        x = self.res2(x)

        # Terceiro upsampling: 8x8 -> 16x16
        x = self.up3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = torch.cat([x, x1], dim=1)  # Concatenar com skip connection
        x = self.res1(x)

        # Camada de saída
        x = self.conv_out(x)
        x = torch.tanh(x)

        return x

class LunarCoreVAE(nn.Module):
    def __init__(self, latent_dim=128, activation=nn.LeakyReLU(0.2)):
        super(LunarCoreVAE, self).__init__()
        self.encoder = LunarCoreEncoder(latent_dim, activation)
        self.decoder = LunarCoreDecoder(latent_dim, activation)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar, skips = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z, skips)
        return recon_x, mean, logvar
