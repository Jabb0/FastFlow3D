import torch
import torch_geometric.nn


class SetConvLayer(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, r: float, sample_rate: float, mlp: torch.nn.Sequential):
        super().__init__()
        self.sample_rate = sample_rate
        self.radius = r
        # see  https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=point%20conv#torch_geometric.nn.conv.PointConv
        self.point_conv = torch_geometric.nn.PointConv(mlp)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.tensor:
        """ Input must be point cloud tensor of shape (n_points, n_features)  """

        # sample points (regions) by using iterative farthest point sampling (FPS)
        idx = torch_geometric.nn.fps(pos, batch, ratio=self.sample_rate)

        # For each region, get all points which are within in the region (defined by radius r)
        row, col = torch_geometric.nn.radius(pos, pos[idx], self.radius, batch, batch[idx])

        edge_index = torch.stack([col, row], dim=0)

        # Apply point net
        x = self.point_conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]

        return x, pos, batch
