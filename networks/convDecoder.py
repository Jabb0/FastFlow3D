import torch


class ConvDecoder(torch.nn.Module):
    """
    Takes four different pillar grid embeddings (obtained from encoder) of two consecutive point clouds pc_prev
    and pc_cur, concatenates them and transform them to a 3D flow vector of shape (batch_size, n_points, 3).

    References
    ----------
    .. Scalable Scene Flow from Point Clouds in the Real World: Philipp Jund and Chris Sweeney
       and Nichola Abdo and Zhifeng Chen and Jonathon Shlens
       https://arxiv.org/pdf/1812.05784.pdf
    """
    def __init__(self):
        super(ConvDecoder, self).__init__()
        # S: Inputs are R, L:
        #               R: (512, 64, 64), concatenation of R_prev and R_cur, which have shape of (256, 64, 64)
        #               L: (256, 128, 128), concatenation of L_prev and L_cur, which have shape of (128, 128, 128)
        # We need 2*256 and 2*128 as input channels, because we feed a concatenation of both embedded point clouds
        self._block_1 = _UpSamplingSkip(
            in_channels_conv_1=2*256, in_channels_conv_2=2*128, out_channels=128, alpha_shape=(64, 64, 256),
            beta_shape=(128, 128, 128))

        # T: Inputs are S, F:
        #               S: (128, 128, 128), output of block_1
        #               F: (128, 256, 256), concatenation of F_prev and F_cur, which have shape of (64, 256, 256)
        # We need 2*64 as input channels, because we feed a concatenation of both embedded point clouds as second arg.
        self._block_2 = _UpSamplingSkip(
            in_channels_conv_1=128, in_channels_conv_2=2*64, out_channels=128, alpha_shape=(128, 128, 128),
            beta_shape=(256, 256, 64))

        # U: Inputs are T, B:
        #               T: (128, 256, 256), output of block_2
        #               B: (128, 512, 512), concatenation of B_prev and B_cur, which have shape of (64, 512, 512)
        # We need 2*64 as input channels, because we feed a concatenation of both embedded point clouds as second arg.
        self._block_3 = _UpSamplingSkip(
            in_channels_conv_1=128, in_channels_conv_2=2*64, out_channels=64, alpha_shape=(256, 256, 128),
            beta_shape=(512, 512, 64))

        # V: Input is U, which is the output of block_3
        self._block_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', padding=(1, 1)),
            # TODO: These two layers are misplaced but seem to decrease the loss.
            #  Create an experiment to test this.
            # torch.nn.BatchNorm2d(64),
            # torch.nn.ReLU(),
        )

    def forward(self, B_prev, F_prev, L_prev, R_prev, B_cur, F_cur, L_cur, R_cur):
        """
        Do the convolutional encoder pass for the input 2D grid embedding.
        :param B_prev: (batch_size, 64, 512, 512) unfiltered 2D grid embedding of previous point cloud.
        :param F_prev: (batch_size, 64, 256, 256) filtered 2D grid embedding of previous point cloud.
        :param L_prev: (batch_size, 128, 128, 128) filtered 2D grid embedding of previous point cloud.
        :param R_prev: (batch_size, 256, 64, 64) filtered 2D grid embedding of previous point cloud.

        :param B_cur: (batch_size, 64, 512, 512) unfiltered 2D grid embedding of current point cloud.
        :param F_cur: (batch_size, 64, 256, 256) filtered 2D grid embedding of current point cloud.
        :param L_cur: (batch_size, 128, 128, 128) filtered 2D grid embedding of current point cloud.
        :param R_cur: (batch_size, 256, 64, 64) filtered 2D grid embedding of current point cloud.

        :return: (batch_size, n_points, 3)
        """
        # Concatenate embedded point clouds
        B = torch.cat((B_prev, B_cur), dim=1)
        R = torch.cat((R_prev, R_cur), dim=1)
        F = torch.cat((F_prev, F_cur), dim=1)
        L = torch.cat((L_prev, L_cur), dim=1)
        # print("After concatenation of both embedded point clouds:")
        # print(f"R Output: {R.shape}")
        # print(f"F Output: {F.shape}")
        # print(f"L Output: {L.shape}")
        # print("-"*12)

        S = self._block_1(R, L)  # S inputs are R, L
        # print(f"S Output: {S.shape}")
        T = self._block_2(S, F)  # T, inputs are S, F
        # print(f"T Output: {T.shape}")
        # print(f"B Output: {B.shape}")
        U = self._block_3(T, B)  # U+V, inputs are T, B
        # print(f"U Output: {U.shape}")
        V = self._block_4(U)
        # print(f"V Output: {V.shape}")

        return V


class _UpSamplingSkip(torch.nn.Module):
    """
    Takes two inputs alpha and beta. Alpha and beta is feed into a convolution, the height and width of
     alpha is scaled up to the height and width of beta by using bilinear interpolation.
     Afterwards alpha and beta are concatenated and fed into two convolutions.
    """
    def __init__(self, in_channels_conv_1, in_channels_conv_2,  out_channels, alpha_shape, beta_shape):
        super(_UpSamplingSkip, self).__init__()
        a_w, a_h, a_d = alpha_shape  # (w, h, d)
        b_w, b_h, b_d = beta_shape  # (w, h, d)
        # D_out = D_in x scale_factor -> D_out / D_in = scale_factor
        # (bring alpha to shape of beta -> alpha is in, beta is out)
        scale_factor = (b_w / a_w, b_h / a_h)

        # U1
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels_conv_1, out_channels=out_channels, kernel_size=(1, 1))
        # U2
        # TODO: Where is the stride of 0.5 that is mentioned in the paper?
        # TODO: Instead of calculating scale_factor one can also give the desired output shape?
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
        U4 = torch.cat((U2, U3), dim=1)
        U5 = self.conv_3(U4)
        U6 = self.conv_4(U5)
        return U6
