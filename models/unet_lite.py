"""
U-Net Lite (Lightweight) Model
=============================
Reduced-parameter U-Net for VRAM-constrained benchmarking.

Architecture changes from standard U-Net++:
  - Base channels reduced from 32->16 (vs U-Net++ 512)
  - Single decoder level (no deep supervision)
  - Total ~1-2M parameters (vs 9M for U-Net++)

Useful for:
  - Testing if heavy architectures are needed for structure
  - Running on 6GB laptops (fits at 256x256 patches)
  - Fast ablation studies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetLite(nn.Module):
    """Lightweight U-Net with reduced channels."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 8)

        # Decoder - use interpolate for proper size matching
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, 2)
        self.dec4 = self._conv_block(base_channels * 8 + base_channels * 8, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)

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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([e4, F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        d3 = self.dec3(torch.cat([e3, F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        d2 = self.dec2(torch.cat([e2, F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        d1 = self.dec1(torch.cat([e1, F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)], dim=1))

        return self.final(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetLite(in_channels=3, out_channels=1).to(device)
    params = count_parameters(model)
    print(f"\nU-Net Lite Parameters: {params:,} ({params/1e6:.2f}M)")

    x = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print("[OK] U-Net Lite created successfully!")