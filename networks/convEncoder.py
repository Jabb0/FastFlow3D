import torch


class ConvEncoder(torch.nn.Module):
    """
    Transforms a 2D pillar embedding into a different size 2D pillar embedding.
    """
    def __init__(self, in_channels=64, out_channels=256):
        super(ConvEncoder).__init__()
        # Input is a 512x512x64 image. Output is a 256x256x64 image.
        self._block_1 = torch.nn.Sequential(
            # C
            torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # D
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # E
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # F
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        # Input is a 256x256x64 image. Output is a 128x128x128 image.
        self._block_2 = torch.nn.Sequential(
            # G
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros'),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # H
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # I
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # J
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # K
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # L
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )

        # Input is a 128x128x128 image. Output is a 64x64x256 image.
        self._block_3 = torch.nn.Sequential(
            # M
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros'),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # N
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # O
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # P
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # Q
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # R
            torch.nn.Conv2d(256, out_channels, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        """
        Do the convolutional encoder pass for the input 2D grid embedding.
        :param x: (batch_size, height, width, channels) of the 2D grid embedding.
        :return: (batch_size, out_height, out_width, )
        """
        block_1 = self._block_1(x)
        block_2 = self._block_2(block_1)
        block_3 = self._block_3(block_2)

        return block_1, block_2, block_3
