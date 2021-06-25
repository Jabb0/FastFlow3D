import torch
from networks.flownet3d.util import make_mlp
from networks.flownet3d.layers import SetConvLayer


class PointFeatureNet(torch.nn.Module):
    """
    PointFeatureNet which is the first part of FlowNet3D and consists of four SetConvLayers

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

        set_conv_mlp_3 = make_mlp(128+3, [128, 128, 256])
        self.set_conv_3 = SetConvLayer(r=2.0, sample_rate=0.25, mlp=set_conv_mlp_3)

        set_conv_mlp_4 = make_mlp(256+3, [256, 256, 512])
        self.set_conv_4 = SetConvLayer(r=4.0, sample_rate=0.25, mlp=set_conv_mlp_4)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        """
        batch_size, n_points, _ = x.shape  # (batch_size, num_points, 3)

        # get features
        features = x[:, :, 3:]
        features = features.view(batch_size * n_points, -1)

        # get pos
        pos = x[:, :, :3]
        pos = pos.view(batch_size * n_points, -1)

        batch = torch.zeros((batch_size, n_points), device=pos.device, dtype=torch.long)
        for i in range(batch_size):
            batch[i] = i
        batch = batch.view(-1)

        features, pos, batch = self.set_conv_1(x=features, pos=pos, batch=batch)
        features, pos, batch = self.set_conv_2(x=features, pos=pos, batch=batch)
        features, pos, batch = self.set_conv_3(x=features, pos=pos, batch=batch)
        features, pos, batch = self.set_conv_4(x=features, pos=pos, batch=batch)
        print("-"*100)
        print(features.shape)
        print(pos.shape)
        print(batch.shape)
        return x