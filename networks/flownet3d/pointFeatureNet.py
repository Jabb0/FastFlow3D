import torch
from networks.flownet3d.util import make_mlp
from networks.flownet3d.layers import SetConvLayer


class PointFeatureNet(torch.nn.Module):
    """
    PointFeatureNet which is the first part of FlowNet3D and consists of four SetConvLayers.

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self, in_channels: int):
        super(PointFeatureNet, self).__init__()

        set_conv_mlp_1 = make_mlp(in_channels, [32, 32, 64])
        self.set_conv_1 = SetConvLayer(r=0.5, sample_rate=0.5, mlp=set_conv_mlp_1)

        set_conv_mlp_2 = make_mlp(64+3, [64, 64, 128])
        self.set_conv_2 = SetConvLayer(r=1.0, sample_rate=0.25, mlp=set_conv_mlp_2)

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """
        Input is a point cloud of shape (batch_size, n_points, n_features),
        where the first three features are the x,y,z coordinate of the point.
        """
        batch_size, n_points, _ = x.shape  # (batch_size, n_points, n_features)
        mask = mask.flatten()

        # get features
        features = x[:, :, 3:]
        features = features.flatten(0, 1)
        features = features[mask, :]

        # get pos
        pos = x[:, :, :3]
        pos = pos.view(batch_size * n_points, -1)
        pos = pos[mask, :]

        batch = torch.arange(batch_size, device=features.device)
        batch = batch.repeat_interleave(n_points)
        batch = batch[mask]

        x1 = (features, pos, batch)
        x2 = self.set_conv_1(x1)
        x3 = self.set_conv_2(x2)

        return x1, x2, x3

#
# from Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule
#
#
# class PointFeatureNetV2(torch.nn.Module):
#     """
#     PointFeatureNet which is the first part of FlowNet3D and consists of four SetConvLayers.
#
#     References
#     ----------
#     .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
#        https://arxiv.org/pdf/1806.01411.pdf
#     """
#     def __init__(self, in_channels: int):
#         super(PointFeatureNetV2, self).__init__()
#         self.n_sample = 8
#
#         self.set_conv_1_2 = PointnetSAModule(
#             mlp=[in_channels - 3, 32, 32, 64],
#             sample_rate=0.5,
#             radius=0.5,
#             nsample=self.n_sample,
#             use_xyz=True,
#         )
#
#         self.set_conv_2_2 = PointnetSAModule(
#             mlp=[64, 64, 64, 128],
#             sample_rate=0.25,
#             radius=1.0,
#             nsample=self.n_sample,
#             use_xyz=True,
#         )
#
#     def forward(self, x: torch.tensor) -> torch.tensor:
#         """
#         Input is a point cloud of shape (batch_size, n_points, n_features),
#         where the first three features are the x,y,z coordinate of the point.
#         """
#         batch_size, n_points, _ = x.shape  # (batch_size, n_points, n_features)
#
#         # get features
#         features = x[:, :, 3:]
#         features = features.permute(0, 2, 1).contiguous()
#
#         # get pos
#         pos = x[:, :, :3].contiguous()
#
#         x1 = (pos, features)
#
#         x2 = self.set_conv_1_2(xyz=pos, features=features)
#         x3 = self.set_conv_2_2(*x2)
#
#         return x1, x2, x3