import torch


class ConvDecoder(torch.nn.Module):
    """
    Transforms a 2D pillar embedding into a different size 2D pillar embedding.
    """
    def __init__(self):
        super(ConvDecoder, self).__init__()
        # S: Input is a 64x64x256 image. Output is a 128x128x128 image.
        self._block_1 = _UpSamplingSkip(
            in_channels_conv_1=256, in_channels_conv_2=128, out_channels=128, alpha_shape=(64, 64, 256),
            beta_shape=(128, 128, 128))

        # T: Input is a 128x128x128 image. Output is a 256x256x128 image.
        self._block_2 = _UpSamplingSkip(
            in_channels_conv_1=128, in_channels_conv_2=64, out_channels=128, alpha_shape=(128, 128, 128), beta_shape=(256, 256, 64))

        # U: Input is a 256x256x128 image. Output is a 512x512x64 image.
        self._block_3 = _UpSamplingSkip(
            in_channels_conv_1=128, in_channels_conv_2=64, out_channels=64, alpha_shape=(256, 256, 128), beta_shape=(512, 512, 64))

        # V
        self._block_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        # TODO Define Ungrid, Concatenate, Linear, Linear

    def forward(self, B, F, L, R):
        """
        Do the convolutional encoder pass for the input 2D grid embedding.
        :param x: (batch_size, height, width, channels) of the 2D grid embedding.
        :return: (batch_size, out_height, out_width, )
        """
        # TODO Concat both embedded point clouds - where?
        S = self._block_1(R, L)  # S inputs are R, L
        print(f"S Output: {S.shape}")
        T = self._block_2(S, F)  # T, inputs are S, F
        print(f"T Output: {T.shape}")
        print(f"B Output: {B.shape}")
        U = self._block_3(T, B)  # U+V, inputs are T, B
        print(f"U Output: {U.shape}")
        V = self._block_4(U)
        print(f"V Output: {V.shape}")

        # TODO Ungrid, Concatenate, Linear, Linear

        return V


class _UpSamplingSkip(torch.nn.Module):
    def __init__(self, in_channels_conv_1, in_channels_conv_2,  out_channels, alpha_shape, beta_shape):
        super(_UpSamplingSkip, self).__init__()

        self.in_channels_conv_1 = in_channels_conv_1
        self.in_channels_conv_2 = in_channels_conv_2
        self.out_channels = out_channels

        self.relu = torch.nn.ReLU()

        a_w, a_h, a_d = alpha_shape  # (w, h, d)
        b_w, b_h, b_d = beta_shape  # (w, h, d)
        # D_out = D_in x scale_factor -> D_out / D_in = scale_factor
        # (bring alpha to shape of beta -> alpha is in, beta is out)
        scale_factor = (b_w / a_w, b_h / a_h)

        # U1
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels_conv_1, out_channels=out_channels, kernel_size=(1, 1))
        # U2
        self.bilinear_interp = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # U3
        self.conv_2 = torch.nn.Conv2d(in_channels=in_channels_conv_2, out_channels=out_channels, kernel_size=(1, 1))
        # U4: concatenation, see forward pass
        # U5
        self.conv_3 = torch.nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                      padding_mode='zeros', padding=(1, 1))
        # U6
        self.conv_4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                      padding_mode='zeros', padding=(1, 1))

    def forward(self, alpha, beta):
        U1 = self.conv_1(alpha)
        U2 = self.bilinear_interp(U1)
        U3 = self.conv_2(beta)
        # TODO Check if concatenation in depth is correct
        U4 = torch.cat((U2, U3), dim=1)
        U5 = self.conv_3(U4)
        U6 = self.conv_4(U5)
        return U6
