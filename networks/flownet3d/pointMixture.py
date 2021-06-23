import torch


class PointMixtureNet(torch.nn.Module):
    """
    PointFeatureNet which is the first part of FlowNet3D and consists of one FlowEmbeddingLayer

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self):
        super(PointMixtureNet, self).__init__()
        # TODO Set FlowEmbeddingLayer

    def forward(self, x):
        """
        """
        return x