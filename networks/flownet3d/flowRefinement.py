import torch
from networks.flownet3d.util import make_mlp
from networks.flownet3d.layers import SetUpConvLayer


class FlowRefinementNet(torch.nn.Module):
    """
    FlowRefinementNet which is the last part of FlowNet3D and consists of four SetUpConvLayers

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self, in_channels: int):
        super(FlowRefinementNet, self).__init__()
        setup_conv_mlp_1 = make_mlp(in_channels, [128, 128, 256])
        self.setup_conv_1 = SetUpConvLayer(r=4.0, mlp=setup_conv_mlp_1)

        setup_conv_mlp_2 = make_mlp(256 + 3, [128, 128, 256])
        self.setup_conv_2 = SetUpConvLayer(r=2.0, mlp=setup_conv_mlp_2)

        setup_conv_mlp_3 = make_mlp(256 + 3, [128, 128, 128])
        self.setup_conv_3 = SetUpConvLayer(r=1.0, mlp=setup_conv_mlp_3)

        setup_conv_mlp_4 = make_mlp(128 + 3, [128, 128, 128])
        self.setup_conv_4 = SetUpConvLayer(r=0.5, mlp=setup_conv_mlp_4)

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
        x = self.setup_conv_1(src=fe_3, target=fe_2)
        x = self.setup_conv_2(src=x, target=pf_curr_3)
        x = self.setup_conv_3(src=x, target=pf_curr_2)
        x = self.setup_conv_4(src=x, target=pf_curr_1)

        return x