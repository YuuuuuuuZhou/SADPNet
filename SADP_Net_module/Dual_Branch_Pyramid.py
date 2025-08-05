import torch
import torch.nn as nn
from torch.nn import functional as F
from Symmetry_Aware_Module import SymmetryAware

class DualBranchPyramid(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.base_channels = base_channels
        self.channels_config = [
            base_channels,
            base_channels // 2,
            base_channels // 4,
            base_channels // 8,
        ]
        self.upsample_x_to_x2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU()
        )
        self.upsample_x2_to_x4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU()
        )
        self.upsample_x4_to_x8 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 8),
            nn.ReLU()
        )

        self.flip_blocks = nn.ModuleList([
            SymmetryAware(channels) for channels in self.channels_config
        ])

        self.fpn_fuse_layers = nn.ModuleList()
        self.fpn_fuse_layers.append(nn.Sequential(
            nn.Conv2d(self.channels_config[3], self.channels_config[2],
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channels_config[2]),
            nn.ReLU(),
            nn.Conv2d(self.channels_config[2] * 2, self.channels_config[2],
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels_config[2])
        ))

        self.fpn_fuse_layers.append(nn.Sequential(
            nn.Conv2d(self.channels_config[2], self.channels_config[1],
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channels_config[1]),
            nn.ReLU(),
            nn.Conv2d(self.channels_config[1] * 2, self.channels_config[1],
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels_config[1])
        ))

        self.fpn_fuse_layers.append(nn.Sequential(
            nn.Conv2d(self.channels_config[1], self.channels_config[0],
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channels_config[0]),
            nn.ReLU(),
            nn.Conv2d(self.channels_config[0] * 2, self.channels_config[0],
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels_config[0])
        ))


    def forward(self, x):
        x_flip = x.flip(-1)

        x2 = self.upsample_x_to_x2(x)
        x2_flip = self.upsample_x_to_x2(x_flip)

        x4 = self.upsample_x2_to_x4(x2)
        x4_flip = self.upsample_x2_to_x4(x2_flip)

        x8 = self.upsample_x4_to_x8(x4)
        x8_flip = self.upsample_x4_to_x8(x4_flip)

        multi_scale_features = [x, x2, x4, x8]
        multi_scale_flips = [x_flip, x2_flip, x4_flip, x8_flip]

        processed_features = []
        for i, flip_block in enumerate(self.flip_blocks):
            try:
                processed = flip_block(multi_scale_features[i], multi_scale_flips[i])
                processed_features.append(processed)
            except Exception as e:
                raise e

        current = processed_features[3]

        for i, fuse_layer in enumerate(self.fpn_fuse_layers):
            target_feat = processed_features[2 - i]

            try:
                downsample_layers = fuse_layer[:4]
                fusion_layer = fuse_layer[4]

                downsampled = current
                for layer in downsample_layers[:-1]:
                    downsampled = layer(downsampled)

                concatenated = torch.cat([downsampled, target_feat], dim=1)

                fused = downsample_layers[-1](concatenated)
                current = fusion_layer(fused)
                current = F.relu(current)

            except Exception as e:
                raise e

        return current