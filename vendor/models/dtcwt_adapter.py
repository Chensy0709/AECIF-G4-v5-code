"""Differentiable DTCWT adapter used only by the Phase 2 DTCWT experiment."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_wavelets import DTCWTForward


class DTCWTResidualAdapter(nn.Module):
    """Low-cost, fixed-filter DTCWT feature adapter with an identity start.

    Input and output are [B, 512, H, W].  The learned pointwise bottleneck is
    transformed by a 3-level, 6-direction DTCWT.  Per-direction magnitudes are
    aligned to the input grid and fused before projecting back to 512 channels.
    The transform is evaluated in float32 (also under AMP) for numerical and
    kernel compatibility; casts remain differentiable.
    """

    def __init__(self, channels: int = 512, bottleneck: int = 8,
                 levels: int = 3, biort: str = "near_sym_a",
                 qshift: str = "qshift_a", mode: str = "symmetric"):
        super().__init__()
        self.channels = int(channels)
        self.bottleneck = int(bottleneck)
        self.levels = int(levels)
        self.reduce = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=True)
        self.transform = DTCWTForward(biort=biort, qshift=qshift, J=levels, mode=mode)
        self.fuse = nn.Sequential(
            nn.Conv2d(bottleneck * 6 * levels, bottleneck, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.expand = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(()))

    def directional_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return aligned DTCWT magnitude features [B, bottleneck*6*J, H, W]."""
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected [B,{self.channels},H,W], got {tuple(x.shape)}")
        batch, _, height, width = x.shape
        # Keep both the projection and transform out of autocast. Conversion is
        # differentiable, and the rest of AECIF-Net may still use AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.reduce(x.float())
            _, highpasses = self.transform(z)
        if highpasses is None or len(highpasses) != self.levels:
            raise RuntimeError("DTCWT returned an unexpected number of levels")
        aligned = []
        for coeff in highpasses:
            if coeff.ndim != 6 or coeff.shape[2] != 6 or coeff.shape[-1] != 2:
                raise RuntimeError(f"Unexpected DTCWT coefficient shape: {tuple(coeff.shape)}")
            # Magnitude is phase invariant and avoids angle discontinuities.
            mag = torch.sqrt(coeff.square().sum(dim=-1) + 1e-12)
            b, c, orientations, h, w = mag.shape
            mag = mag.reshape(b, c * orientations, h, w)
            if (h, w) != (height, width):
                mag = F.interpolate(mag, size=(height, width), mode="bilinear",
                                    align_corners=False)
            aligned.append(mag)
        return torch.cat(aligned, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source_dtype = x.dtype
        features = self.directional_features(x)
        features = features.to(dtype=source_dtype)
        residual = self.expand(self.fuse(features))
        return x + self.alpha.to(dtype=x.dtype) * residual
