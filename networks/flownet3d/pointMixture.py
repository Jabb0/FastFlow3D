from typing import Tuple

import torch

from networks.flownet3d.layers import SetConvLayer, FlowEmbeddingLayer


class PointMixtureNet(torch.nn.Module):
    """
    PointMixtureNet which is the second part of FlowNet3D and consists
    of one FlowEmbeddingLayer and two SetConvLayers.

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self, n_samples: int = 2):
        super(PointMixtureNet, self).__init__()
        self.n_samples = n_samples
        self.fe_1 = FlowEmbeddingLayer(mlp=[(2*128), 128, 128, 128],
                                     sample_rate=1.0,
                                     radius=5.0,
                                     n_samples=self.n_samples,
                                     use_xyz=True
                                     )

        self.set_conv_1 = SetConvLayer(
            mlp=[128, 128, 128, 256],
            sample_rate=0.25,
            radius=2.0,
            n_samples=self.n_samples,
            use_xyz=True,
        )

        self.set_conv_2 = SetConvLayer(
            mlp=[256, 256, 256, 512],
            sample_rate=0.25,
            radius=4.0,
            n_samples=self.n_samples,
            use_xyz=True,
        )

    def forward(self, x1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                x2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.tensor:
        """
        Inputs are two point clouds, each point cloud is a tuple (features, pos, batch).
        Both point clouds are combined by using the FlowEmbeddingLayer and afterwards the combined representation is
        down-sampled by two SetConvLayers.
        """
        fe_1 = self.fe_1(*x1, *x2)
        fe_2 = self.set_conv_1(*fe_1)
        fe_3 = self.set_conv_2(*fe_2)

        return fe_1, fe_2, fe_3

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from typing import List

def make_mlp(in_channels: int, mlp_channels: List[int], batch_norm: bool = True):
    """ Creates a neural network with one linear layer for each entry in mlp_channels
        in_channels is an integer, which defines the number of channels of the input to the neural network.
        mlp_channels is a list of ints, where each int denotes the output features of the corresponding linear layer.
     """
    assert len(mlp_channels) >= 1
    layers = []

    for c in mlp_channels:
        layers += [Linear(in_channels, c)]
        if batch_norm:
            layers += [BatchNorm1d(c)]
        layers += [ReLU()]

        in_channels = c

    return Sequential(*layers)


class PointMixtureNetV2(torch.nn.Module):
    """
    PointMixtureNet which is the second part of FlowNet3D and consists
    of one FlowEmbeddingLayer and two SetConvLayers.
    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self):
        super(PointMixtureNetV2, self).__init__()
        # 2*128+3, because we have f_i, g_j, p_i - p_j
        fe_mlp_1 = make_mlp(2*128+3, [128, 128, 128])
        self.fe_1 = FlowEmbeddingLayerV2(r=5.0, sample_rate=1.0, mlp=fe_mlp_1, max_num_neighbors=64)

        set_conv_mlp_1 = make_mlp(128+3, [128, 128, 256])
        self.set_conv_1 = SetConvLayerV2(r=2.0, sample_rate=0.25, mlp=set_conv_mlp_1, max_num_neighbors=8)

        set_conv_mlp_2 = make_mlp(256+3, [256, 256, 512])
        self.set_conv_2 = SetConvLayerV2(r=4.0, sample_rate=0.25, mlp=set_conv_mlp_2, max_num_neighbors=8)

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