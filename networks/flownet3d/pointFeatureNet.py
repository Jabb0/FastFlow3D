import torch
from util import make_mlp
from layers import SetConvLayer


class PointFeatureNet(torch.nn.Module):
    """
    PointFeatureNet which is the first part of FlowNet3D and consists of four SetConvLayers

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self):
        super(PointFeatureNet, self).__init__()

        set_conv_mlp_1 = make_mlp(8, [32, 32, 64])
        self.set_conv_1 = SetConvLayer(sample_rate=0.5, radius=0.5, mlp=set_conv_mlp_1)

        set_conv_mlp_2 = make_mlp(64, [64, 64, 128])
        self.set_conv_2 = SetConvLayer(sample_rate=0.25, radius=1.0, mlp=set_conv_mlp_2)

        set_conv_mlp_3 = make_mlp(128, [128, 128, 256])
        self.set_conv_3 = SetConvLayer(sample_rate=0.25, radius=2.0, mlp=set_conv_mlp_3)

        set_conv_mlp_4 = make_mlp(256, [256, 256, 512])
        self.set_conv_4 = SetConvLayer(sample_rate=0.25, radius=4.0, mlp=set_conv_mlp_4)

    def forward(self, x):
        """
        """
        return x