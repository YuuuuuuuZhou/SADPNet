import torch
import torch.nn as nn
from torch.nn import functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.attn(avg_pool + max_pool)


class SymmetryAware(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.context_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.diff_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            SpatialAttention()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x, x_flip):
        context = self.context_conv(x)

        gate = self.diff_gate(torch.cat([x, x_flip], dim=1))
        diff = (x - x_flip) * gate

        edges = self.edge_conv(diff)

        fused = torch.cat([context, x_flip, edges], dim=1)
        return F.relu(x + self.fusion(fused))

