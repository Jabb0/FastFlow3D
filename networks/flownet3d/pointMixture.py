import torch
from typing import Tuple
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
