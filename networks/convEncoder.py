import torch


class ConvEncoder(torch.nn.Module):
    """
    Transforms a 2D pillar embedding into a 3D flow vector
    """
    def __init__(self, in_channels=64, out_channels=256):
        padding = (1, 1)
        super(ConvEncoder, self).__init__()
        # Input is a 512x512x64 image. Output is a 256x256x64 image.
        self._block_1 = torch.nn.Sequential(
            # C: (1, 64, 256, 256)
            torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # D: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # E: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # F: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        # Input is a 256x256x64 image. Output is a 128x128x128 image.
        self._block_2 = torch.nn.Sequential(
            # G: (1, 128, 128, 128)
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # H: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # I: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # J: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # K: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # L: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )

        # Input is a 128x128x128 image. Output is a 64x64x256 image.
        self._block_3 = torch.nn.Sequential(
            # M: (1, 256, 64, 64)
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # N: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # O: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # P: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # Q: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # R: (1, 256, 64, 64)
            torch.nn.Conv2d(256, out_channels, kernel_size=(3, 3), stride=(1, 1),
                            padding_mode='zeros', padding=padding),
            torch.nn.BatchNorm2d(out_channels),
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
