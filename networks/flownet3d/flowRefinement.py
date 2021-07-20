import torch

from networks.flownet3d.layers import SetConvUpLayer


class FlowRefinementNet(torch.nn.Module):
    """
    FlowRefinementNet which is the last part of FlowNet3D and consists of four SetUpConvLayers

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self, in_channels: int, n_samples: int = 2):
        super(FlowRefinementNet, self).__init__()
        self.n_samples = n_samples

        self.setup_conv_1 = SetConvUpLayer(
            mlp=[in_channels, 128, 128, 256],
            sample_rate=4.0,
            radius=4.0,
            n_samples=self.n_samples,
            use_xyz=True,
        )

        self.setup_conv_2 = SetConvUpLayer(
            mlp=[256, 128, 128, 256],
            sample_rate=4.0,
            radius=2.0,
            n_samples=self.n_samples,
            use_xyz=True,
        )

        self.setup_conv_3 = SetConvUpLayer(
            mlp=[256, 128, 128, 128],
            sample_rate=4.0,
            radius=1.0,
            n_samples=self.n_samples,
            use_xyz=True,
        )

        self.setup_conv_4 = SetConvUpLayer(
            mlp=[128, 128, 128, 128],
            sample_rate=4.0,
            radius=0.5,
            n_samples=self.n_samples,
            use_xyz=True,
        )

    def forward(self, pf_curr_1: torch.tensor, pf_curr_2: torch.tensor,
                pf_curr_3: torch.tensor, fe_2: torch.tensor, fe_3: torch.tensor) -> torch.tensor:
        """
        Propagate features to the original point cloud points
         Propagate:
            1. from fe_3 to fe_2
            2. from output of 1. to pf_curr_3
            3. from output of 2. to pf_curr_2
            4. from output of 3. to pf_curr_1
        """
        # target: Are locations we want to propagate the source point features to
        # target must have higher number of points than source
        x = self.setup_conv_1(*fe_3, *fe_2)
        x = self.setup_conv_2(*x, *pf_curr_3)
        x = self.setup_conv_3(*x, *pf_curr_2)
        x = self.setup_conv_4(*x, *pf_curr_1)

        return x