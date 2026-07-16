"""
Attention U-Net Baseline Model
==============================
Attention U-Net (Oktay et al., 2018) - Efficient attention-gated segmentation.

Key features:
  - Attention gates at decoder skip connections  
  - Focuses on relevant features, suppresses background
  - ~4-5M parameters (lighter than U-Net++)
  - Fits in 4GB VRAM on 128x128 patches

Recommended for baseline comparison in Paper 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention gate for skip connections."""

    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 4

        self.conv_gating = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_skip, gating_signal):
        theta_x = self.conv_skip(x_skip)
        # Interpolate gating to match skip connection spatial size
        phi_g = self.conv_gating(F.interpolate(gating_signal, size=x_skip.shape[2:], mode='bilinear', align_corners=False))
        f = self.relu(theta_x + phi_g)
        attention = self.psi(f)
        return x_skip * attention


class AttentionUNet(nn.Module):
    """Attention U-Net architecture."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        self.base_channels = base_channels

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # Decoder with attention
        self.att4 = AttentionBlock(base_channels * 8, base_channels * 16)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = self._conv_block(base_channels * 16, base_channels * 8)

        self.att3 = AttentionBlock(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)

        self.att2 = AttentionBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)

        self.att1 = AttentionBlock(base_channels, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Decoder with attention (upsample first, then concatenate with original skip)
        d4 = self.dec4(torch.cat([self.att4(e4, b), self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([self.att3(e3, d4), self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([self.att2(e2, d3), self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([self.att1(e1, d2), self.up1(d2)], dim=1))

        return self.final(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    params = count_parameters(model)
    print(f"\nAttention U-Net Parameters: {params:,} ({params/1e6:.1f}M)")

    x = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print("[OK] Attention U-Net created successfully!")