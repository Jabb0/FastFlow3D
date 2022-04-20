"""
MIT License

Copyright (c) 2021 Felix (Jabb0), Aron (arndz), Carlos (cmaranes)
Source: https://github.com/Jabb0/FastFlow3D

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch


class ConvEncoder(torch.nn.Module):
    """
    Transforms a 2D pillar embedding into a 3D flow vector
    """

    def __init__(self, in_channels=64, out_channels=256, use_group_norm=False, group_norm_groups=4):
        """

        This class can either use batch norm or group norm.
            GroupNorm has LayerNorm and InstanceNorm as special cases.
                It allows to work with much smaller batch sizes.
        :param in_channels:
        :param out_channels:
        :param use_group_norm:
        """
        padding = (1, 1)
        super(ConvEncoder, self).__init__()
        # Input is a 512x512x64 image. Output is a 256x256x64 image.
        # (512input - 3kernel + 2 * 1pad) / 2stride + 1 = 256.5 = 256
        # The correct output is only achieved with padding. This yields for all the conv layers.
        self._block_1 = torch.nn.Sequential(
            # C: (1, 64, 256, 256)
            torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(64) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 64),
            torch.nn.ReLU(),
            # D: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(64) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 64),
            torch.nn.ReLU(),
            # E: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(64) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 64),
            torch.nn.ReLU(),
            # F: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(64) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 64),
            torch.nn.ReLU(),
        )

        # Input is a 256x256x64 image. Output is a 128x128x128 image.
        self._block_2 = torch.nn.Sequential(
            # G: (1, 128, 128, 128)
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 128),
            torch.nn.ReLU(),
            # H: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 128),
            torch.nn.ReLU(),
            # I: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 128),
            torch.nn.ReLU(),
            # J: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 128),
            torch.nn.ReLU(),
            # K: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 128),
            torch.nn.ReLU(),
            # L: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 128),
            torch.nn.ReLU(),
        )

        # Input is a 128x128x128 image. Output is a 64x64x256 image.
        self._block_3 = torch.nn.Sequential(
            # M: (1, 256, 64, 64)
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 256),
            torch.nn.ReLU(),
            # N: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 256),
            torch.nn.ReLU(),
            # O: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 256),
            torch.nn.ReLU(),
            # P: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 256),
            torch.nn.ReLU(),
            # Q: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256) if not use_group_norm else torch.nn.GroupNorm(group_norm_groups, 256),
            torch.nn.ReLU(),
            # R: (1, 256, 64, 64)
            torch.nn.Conv2d(256, out_channels, kernel_size=(3, 3), stride=(1, 1),
                            padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(out_channels) if not use_group_norm
            else torch.nn.GroupNorm(group_norm_groups, out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, grid):
        """
        Do the convolutional encoder pass for the input 2D grid embedding.
        :param grid: (batch_size, channels, height, width) of the 2D grid embedding.
        :return: (batch_size, channels, out_height, out_width)
        """
        # print(f"Input shape {grid.shape}")
        # Block 1 -> from C to F
        F = self._block_1(grid)  # Output: F
        # print(f"F Output: {F.shape}")
        # Block 2 -> from G to L
        L = self._block_2(F)  # Output: L
        # print(f"L Output: {L.shape}")
        # Block 3 -> from M to R
        R = self._block_3(L)  # Output: R
        # print(f"R: {R.shape}")

        return F, L, R
