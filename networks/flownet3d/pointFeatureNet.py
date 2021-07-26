import torch

from networks.flownet3d.layers import SetConvLayer


class PointFeatureNet(torch.nn.Module):
    """
    PointFeatureNet which is the first part of FlowNet3D and consists of four SetConvLayers.

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self, in_channels: int, n_samples: int = 2):
        super(PointFeatureNet, self).__init__()
        self.n_samples = n_samples

        self.set_conv_1_2 = SetConvLayer(
            mlp=[in_channels - 3, 32, 32, 64],
            sample_rate=0.5,
            radius=0.5,
            n_samples=self.n_samples,
            use_xyz=True,
        )

        self.set_conv_2_2 = SetConvLayer(
            mlp=[64, 64, 64, 128],
            sample_rate=0.25,
            radius=1.0,
            n_samples=self.n_samples,
            use_xyz=True,
        )

    def forward(self, x: torch.tensor, features) -> torch.tensor:
        """
        Input is a point cloud of shape (batch_size, n_points, n_features),
        where the first three features are the x,y,z coordinate of the point.
        """
        batch_size, n_points, _ = x.shape  # (batch_size, n_points, n_features)

        # get features
        features = features.permute(0, 2, 1).contiguous()

        # get pos
        pos = x[:, :, :3].contiguous()

        x1 = (pos, features)

        x2 = self.set_conv_1_2(*x1)
        x3 = self.set_conv_2_2(*x2)

        return x1, x2, x3