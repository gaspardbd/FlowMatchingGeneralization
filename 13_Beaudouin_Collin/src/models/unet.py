"""Small UNet for image-scale flow matching, trainable on a free Colab T4."""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time variable."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + x


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.res = ResBlock(in_ch, time_emb_dim)
        self.conv_down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        h = self.res(x, t_emb)
        return self.conv_down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.res = ResBlock(out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.res(x, t_emb)


class SmallUNet(nn.Module):
    """Minimal UNet (3 levels) for 28x28 or 32x32 single-channel images (~1.5M params)."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, time_emb_dim: int = 64):
        super().__init__()
        c = base_channels
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, c, 3, padding=1)
        self.down1 = DownBlock(c, c * 2, time_emb_dim)
        self.down2 = DownBlock(c * 2, c * 4, time_emb_dim)

        self.bottleneck = ResBlock(c * 4, time_emb_dim)

        self.up2 = UpBlock(c * 4, c * 2, c * 2, time_emb_dim)
        self.up1 = UpBlock(c * 2, c, c, time_emb_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.conv_in(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x = self.bottleneck(x, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        return self.conv_out(x)
