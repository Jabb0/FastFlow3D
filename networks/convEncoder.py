import torch


class ConvEncoder(torch.nn.Module):
    """
    Transforms a 2D pillar embedding into a different size 2D pillar embedding.
    """
    def __init__(self, in_channels=64, out_channels=256):
        super(ConvEncoder, self).__init__()
        # Input is a 512x512x64 image. Output is a 256x256x64 image.
        self._block_1 = torch.nn.Sequential(
            # C: (1, 64, 256, 256)
            torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # D: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # E: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # F: (1, 64, 256, 256)
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        # Input is a 256x256x64 image. Output is a 128x128x128 image.
        self._block_2 = torch.nn.Sequential(
            # G: (1, 128, 128, 128)
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # H: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # I: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # J: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # K: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # L: (1, 128, 128, 128)
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )

        # Input is a 128x128x128 image. Output is a 64x64x256 image.
        self._block_3 = torch.nn.Sequential(
            # M: (1, 256, 64, 64)
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # N: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # O: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # P: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # Q: (1, 256, 64, 64)
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # R: (1, 256, 64, 64)
            torch.nn.Conv2d(256, out_channels, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

        # S: Input is a 64x64x256 image. Output is a 128x128x128 image.
        self._block_4 = _UpSamplingSkip(
            in_channels=256, out_channels=128, alpha_shape=(64, 64, 256), beta_shape=(128, 128, 128))

        # T: Input is a 128x128x128 image. Output is a 256x256x128 image.
        self._block_5 = _UpSamplingSkip(
            in_channels=128, out_channels=128, alpha_shape=(128, 128, 128), beta_shape=(256, 256, 64))

        # U: Input is a 256x256x128 image. Output is a 512x512x64 image.
        self._block_6 = _UpSamplingSkip(
            in_channels=128, out_channels=64, alpha_shape=(256, 256, 128), beta_shape=(512, 512, 64))

        # V
        self._block_7 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

        # TODO Ungrid, Concatenate, Linear, Linear

    def forward(self, x):
        """
        Do the convolutional encoder pass for the input 2D grid embedding.
        :param x: (batch_size, height, width, channels) of the 2D grid embedding.
        :return: (batch_size, out_height, out_width, )
        """
        block_1 = self._block_1(x)  # Output: F
        block_2 = self._block_2(block_1)  # Output: L
        block_3 = self._block_3(block_2)  # Output: R
        print(f"Block 3 Output: {block_3.shape}")
        block_4 = self._block_4(block_3, block_2)  # S inputs are R, L
        print(f"Block 4 Output: {block_4.shape}")
        block_5 = self._block_5(block_4, block_1)  # T, inputs are S, F
        print(f"Block 5 Output: {block_5.shape}")
        block_6 = self._block_6(block_5, x)  # U+V, inputs are T, B
        print(f"Block 6 Output: {block_6.shape}")
        block_7 = self._block_7(block_6)
        print(f"Block 6 Output: {block_7.shape}")

        return block_1, block_2, block_3


class _UpSamplingSkip(torch.nn.Module):
    def __init__(self, in_channels,  out_channels, alpha_shape, beta_shape):
        super(_UpSamplingSkip, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = torch.nn.ReLU()

        a_w, a_h, a_d = alpha_shape  # (w, h, d)
        b_w, b_h, b_d = beta_shape  # (w, h, d)
        # D_out = D_in x scale_factor -> D_out / D_in = scale_factor
        # (bring alpha to shape of beta -> alpha is in, beta is out)
        scale_factor = (b_w / a_w, b_h / a_h, b_d / a_d)

        # U1
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        # U2
        self.bilinear_interp = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        # U3
        self.conv_2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        # U4: concatenation, see forward pass
        # U5
        self.conv_3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3))
        # U6
        self.conv_4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3))

    def forward(self, alpha, beta):
        U1 = self.conv_1(alpha)
        U2 = self.bilinear_interp(U1)
        U3 = self.conv_2(beta)
        # TODO concat
        # U4 = torch.cat(U2, U3)
        U5 = self.con3(U4)
        U6 = self.conv_4(U5)
        return alpha
