import torch
from networks.flownet3d.util import make_mlp
from networks.flownet3d.layers import SetConvLayer, FlowEmbeddingLayer
from typing import Tuple


class PointMixtureNet(torch.nn.Module):
    """
    PointMixtureNet which is the second part of FlowNet3D and consists
    of one FlowEmbeddingLayer and two SetConvLayers.

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self):
        super(PointMixtureNet, self).__init__()
        # 2*128+3, because we have f_i, g_j, p_i - p_j
        fe_mlp_1 = make_mlp(2*128+3, [128, 128, 128])
        self.fe_1 = FlowEmbeddingLayer(r=5.0, sample_rate=1.0, mlp=fe_mlp_1)

        set_conv_mlp_1 = make_mlp(128+3, [128, 128, 256])
        self.set_conv_1 = SetConvLayer(r=2.0, sample_rate=0.25, mlp=set_conv_mlp_1)

        set_conv_mlp_2 = make_mlp(256+3, [256, 256, 512])
        self.set_conv_2 = SetConvLayer(r=4.0, sample_rate=0.25, mlp=set_conv_mlp_2)

    def forward(self, x1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                x2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.tensor:
        """
        Inputs are two point clouds, each point cloud is a tuple (features, pos, batch).
        Both point clouds are combined by using the FlowEmbeddingLayer and afterwards the combined representation is
        down-sampled by two SetConvLayers.
        """
        fe_1 = self.fe_1(x1, x2)
        fe_2 = self.set_conv_1(fe_1)
        fe_3 = self.set_conv_2(fe_2)

        return fe_1, fe_2, fe_3
