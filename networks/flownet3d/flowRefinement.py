import torch


class FlowRefinementNet(torch.nn.Module):
    """
    PointFeatureNet which is the first part of FlowNet3D and consists of four SetUpConvLayers

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self):
        super(FlowRefinementNet, self).__init__()
        # TODO: Set 4 SetUpConvLayers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        """
        return x