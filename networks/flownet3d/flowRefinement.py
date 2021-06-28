import torch
from networks.flownet3d.util import make_mlp
from networks.flownet3d.layers import SetUpConvLayer


class FlowRefinementNet(torch.nn.Module):
    """
    PointFeatureNet which is the first part of FlowNet3D and consists of four SetUpConvLayers

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self, in_channels: int):
        super(FlowRefinementNet, self).__init__()
        setup_conv_mlp_1 = make_mlp(in_channels, [128, 128, 256])
        self.setup_conv_1 = SetUpConvLayer(r=4.0, sample_rate=4, mlp=setup_conv_mlp_1)

        setup_conv_mlp_2 = make_mlp(in_channels, [128, 128, 256])
        self.setup_conv_2 = SetUpConvLayer(r=2.0, sample_rate=4, mlp=setup_conv_mlp_2)

        setup_conv_mlp_3 = make_mlp(in_channels, [128, 128, 128])
        self.setup_conv_3 = SetUpConvLayer(r=1.0, sample_rate=4, mlp=setup_conv_mlp_3)

        setup_conv_mlp_4 = make_mlp(in_channels, [128, 128, 128])
        self.setup_conv_4 = SetUpConvLayer(r=0.5, sample_rate=2, mlp=setup_conv_mlp_4)

    def forward(self, src: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        """
        x = self.setup_conv_1(src=src, target=target)
        print(src[0].shape)
        print(x[0].shape)

        return src