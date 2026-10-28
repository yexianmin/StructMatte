"""StructMatte's PSE, cross-view consistency loss, and CGSF modules.

The architectures, view sampling, weighted summations, and nonlinear structural
projection reproduce the original inference pathway. Module paths are renamed
for agreement with the paper; legacy checkpoints require key migration.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        groups = math.gcd(in_channels, out_channels)
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )



class ProgressiveStructureEncoder(nn.Module):
    """PSE: three independent encoders, with three levels per input view."""

    def __init__(self):
        super().__init__()
        self.downsample_x2 = nn.Upsample(scale_factor=0.5, mode="nearest")
        self.view1_level1 = nn.Sequential(
            DoubleConv(4, 4),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, groups=4),
        )
        self.view1_level2 = nn.Sequential(
            DoubleConv(8, 8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, groups=4),
        )
        self.view1_level3 = nn.Sequential(
            DoubleConv(16, 16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, groups=4),
        )
        self.view2_level1 = nn.Sequential(
            DoubleConv(4, 4),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, groups=4),
        )
        self.view2_level2 = nn.Sequential(
            DoubleConv(8, 8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, groups=4),
        )
        self.view2_level3 = nn.Sequential(
            DoubleConv(16, 16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, groups=4),
        )
        self.view4_level1 = nn.Sequential(
            DoubleConv(4, 4),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, groups=4),
        )
        self.view4_level2 = nn.Sequential(
            DoubleConv(8, 8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, groups=4),
        )
        self.view4_level3 = nn.Sequential(
            DoubleConv(16, 16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, groups=4),
        )

    def forward(self, image_trimap):
        view1 = image_trimap
        view2 = self.downsample_x2(view1)
        view4 = self.downsample_x2(view2)
        representations = []
        for level in (1, 2, 3):
            view1 = getattr(self, f"view1_level{level}")(view1)
            view2 = getattr(self, f"view2_level{level}")(view2)
            view4 = getattr(self, f"view4_level{level}")(view4)
            representations.append((view1, view2, view4))
        return tuple(representations)


class CrossViewStructureConsistencyLoss(nn.Module):
    """CSC Loss: average of all three view-pair MSEs at all three levels."""

    def __init__(self):
        super().__init__()
        self.upsample_x2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, representations):
        per_level = []
        for fine, medium, coarse in representations:
            pairwise_mse = (
                F.mse_loss(fine, self.upsample_x2(medium))
                + F.mse_loss(fine, self.upsample_x4(coarse))
                + F.mse_loss(medium, self.upsample_x2(coarse))
            ) / 3
            per_level.append(pairwise_mse)
        return sum(per_level) / len(per_level)


class CrossLevelStructureFusion(nn.Module):
    """CSF: fine-to-coarse aggregation, followed by single-channel projection."""

    def __init__(self):
        super().__init__()
        self.level1 = nn.Sequential(
            DoubleConv(8, 8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, groups=4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.level2 = nn.Sequential(
            DoubleConv(16, 16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False, groups=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.level3 = nn.Sequential(
            DoubleConv(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False, groups=4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.projection = nn.Sequential(
            DoubleConv(64, 32),
            DoubleConv(32, 16),
            DoubleConv(16, 8),
            DoubleConv(8, 4),
            DoubleConv(4, 1),
        )

    def forward(self, level1, level2, level3):
        aggregated = self.level1(level1) + level2
        aggregated = self.level2(aggregated) + level3
        aggregated = self.level3(aggregated)
        return self.projection(aggregated)


class ConsistencyGuidedStructureFusion(nn.Module):
    """CGSF: weighted cross-view fusion followed by CSF."""

    def __init__(self):
        super().__init__()
        self.upsample_x2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode="bilinear")
        # Fixed inverse-square weights from the paper (not learnable).
        self.fusion_weights = (16 / 21, 4 / 21, 1 / 21)
        self.csf = CrossLevelStructureFusion()
        # Names of learnable parameters correspond to a, gamma and beta.
        self.input_scale = nn.Parameter(torch.tensor(0.5))   # a
        self.output_scale = nn.Parameter(torch.tensor(0.5))  # gamma
        self.bias = nn.Parameter(torch.tensor(0.1))          # beta

    def forward(self, representations):
        fused_levels = []
        w1, w2, w4 = self.fusion_weights
        for view1, view2, view4 in representations:
            fused_levels.append(
                w1 * view1
                + w2 * self.upsample_x2(view2)
                + w4 * self.upsample_x4(view4)
            )
        projected = self.csf(*fused_levels)
        return self.output_scale * torch.tanh(self.input_scale * projected) + self.bias


class StructureEnhancementBranch(nn.Module):
    """Structure branch consisting of PSE, CSC Loss and CGSF."""

    def __init__(self):
        super().__init__()
        self.pse = ProgressiveStructureEncoder()
        self.csc_loss = CrossViewStructureConsistencyLoss()
        self.cgsf = ConsistencyGuidedStructureFusion()

    def forward(self, image_trimap):
        representations = self.pse(image_trimap)
        structural_guidance = self.cgsf(representations)
        if self.training:
            return structural_guidance, self.csc_loss(representations)
        return structural_guidance
