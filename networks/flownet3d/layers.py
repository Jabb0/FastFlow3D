import torch
import torch_geometric.nn

from typing import Tuple, Union
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor, Adj
from torch_geometric.nn.conv import MessagePassing


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

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Input must be point cloud tensor of shape (n_points, n_features)  """
        features, pos, batch = x

        # sample points (regions) by using iterative farthest point sampling (FPS)
        idx = torch_geometric.nn.fps(pos, batch, ratio=self.sample_rate)

        # For each region, get all points which are within in the region (defined by radius r)
        row, col = torch_geometric.nn.radius(pos, pos[idx], self.radius, batch, batch[idx])
        # TODO: This stack is unecessary the .radius function already returns the desired index
        #   Please check if this is true or we need to unsqueeze a batch dimension
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
        self.point_conv = _FlowEmbeddingPointConv(mlp)

    def forward(self, x1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                x2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1_features, x1_pos, x1_batch = x1
        x2_features, x2_pos, x2_batch = x2

        # For each points in x2, find all points in x, within distance of r
        row, col = torch_geometric.nn.radius(x1_pos, x2_pos, self.radius, x1_batch, x2_batch)
        edge_index = torch.stack([col, row], dim=0)  # build COO format matrix

        x = self.point_conv((x1_features, x2_features), (x1_pos, x2_pos), edge_index)
        pos, batch = x2_pos, x2_batch

        return x, pos, batch


class SetUpConvLayer(torch.nn.Module):
    """
    TODO
    """
    def __init__(self, r: float, mlp: torch.nn.Sequential):
        super().__init__()
        self.radius = r
        # add self loops must be false, because we only aggregate features of points in the target input
        # and we not include the point of the source input
        self.point_conv = torch_geometric.nn.PointConv(mlp, add_self_loops=False)

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_features, src_pos, src_batch = src  # embeddings
        target_features, target_pos, target_batch = target

        # For each region, get all points which are within in the region (defined by radius r)
        # row, col = torch_geometric.nn.radius(target_pos, src_pos, self.radius, target_batch, src_batch)
        row, col = torch_geometric.nn.radius(src_pos, target_pos, self.radius, src_batch, target_batch)
        edge_index = torch.stack([col, row], dim=0)

        # Apply point net
        # features = self.point_conv(target_features, (target_pos, src_pos), edge_index)
        features = self.point_conv((src_features, target_features), (src_pos, target_pos), edge_index)
        pos, batch = target_pos, target_batch

        return features, pos, batch


class _FlowEmbeddingPointConv(MessagePassing):
    def __init__(self, mlp: torch.nn.Sequential, aggr: str = 'max'):
        super(_FlowEmbeddingPointConv, self).__init__(aggr=aggr)
        self.nn = mlp
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[torch.Tensor, PairTensor], edge_index: Adj) -> torch.Tensor:
        """"""
        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, torch.Tensor):
            pos: PairTensor = (pos, pos)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, pos_i: torch.Tensor, pos_j: torch.Tensor) -> torch.Tensor:
        msg = torch.cat([x_i, x_j, pos_j - pos_i], dim=1)
        msg = self.nn(msg)
        return msg
