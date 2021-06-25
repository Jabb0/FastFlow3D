import torch
import torch_geometric.nn

from typing import Tuple


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

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.tensor:
        """ Input must be point cloud tensor of shape (n_points, n_features)  """
        features, pos, batch = x

        # sample points (regions) by using iterative farthest point sampling (FPS)
        idx = torch_geometric.nn.fps(pos, batch, ratio=self.sample_rate)

        # For each region, get all points which are within in the region (defined by radius r)
        row, col = torch_geometric.nn.radius(pos, pos[idx], self.radius, batch, batch[idx])

        edge_index = torch.stack([col, row], dim=0)

        # Apply point net
        features = self.point_conv(features, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]

        return features, pos, batch


class FlowEmbeddingLayer(torch.nn.Module):
    """
    TODO
    """
    def __init__(self, r: float, sample_rate: float, mlp: torch.nn.Sequential):
        super().__init__()
        self.sample_rate = sample_rate
        self.radius = r
        self.point_conv = torch_geometric.nn.PointConv(mlp)
        self.n_knns = 1

    def forward(self, x1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                x2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.tensor:
        x1_features, x1_pos, x1_batch = x1
        x2_features, x2_pos, x2_batch = x2

        row, col = torch_geometric.nn.knn(x1_pos, x2_pos, self.n_knns, x1_batch, x2_batch)
        edge_index = torch.stack([col, row], dim=0)

        x = self.point_conv((x1_features, x2_features), (x1_pos, x2_pos), edge_index)
        pos, batch = x2_pos, x2_batch

        return x, pos, batch
