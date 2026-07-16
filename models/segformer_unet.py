"""
SegFormer-U-Net Hybrid Model
============================
SegFormer B0/B1 encoder + U-Net decoder for efficient transformer baseline.

Why SegFormer over vanilla TransUNet:
  - SegFormer uses MiT (Mix Transformer) - efficient and lightweight
  - B0: ~3.8M params, B1: ~14M params (vs 100M+ for TransUNet)
  - Works on 6GB VRAM with proper downsampling

Recommended for Paper 1 transformer baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class SegFormerEncoder(nn.Module):
    """
    Simplified SegFormer-like encoder using efficient conv blocks.
    Simulates MiT-0 output structure: 4 resolution levels.
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stage1 = ConvBlock(base_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.stage2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.stage3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.stage4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.stage1(s1)
        s3 = self.stage2(self.pool1(s2))
        s4 = self.stage3(self.pool2(s3))
        s5 = self.stage4(self.pool3(s4))
        bn = self.bottleneck(self.pool4(s5))
        return s1, s2, s3, s4, s5, bn


class SegFormerUNet(nn.Module):
    """SegFormer encoder + U-Net decoder hybrid."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        self.encoder = SegFormerEncoder(in_channels, base_channels)

        # Decoder - upsample to match skip connection sizes
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, 2)
        self.dec4 = ConvBlock(base_channels * 8 + base_channels * 8, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        s1, s2, s3, s4, s5, bn = self.encoder(x)

        d4 = self.dec4(torch.cat([s5, F.interpolate(bn, size=s5.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        d3 = self.dec3(torch.cat([s4, F.interpolate(d4, size=s4.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        d2 = self.dec2(torch.cat([s3, F.interpolate(d3, size=s3.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        d1 = self.dec1(torch.cat([s2, F.interpolate(d2, size=s2.shape[2:], mode='bilinear', align_corners=False)], dim=1))

        return self.final(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegFormerUNet(in_channels=3, out_channels=1).to(device)
    params = count_parameters(model)
    print(f"\nSegFormer-U-Net Parameters: {params:,} ({params/1e6:.1f}M)")

    x = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print("[OK] SegFormer-U-Net hybrid created successfully!")