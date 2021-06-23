import torch
from torch_geometric.nn import fps, radius, PointConv


class SetConvLayer(torch.nn.Module):
    """
    TODO
    """
    def __init__(self, sample_rate, radius, mlp):
        super().__init__()
        self.sample_rate = sample_rate
        self.radius = radius
        self.point_conv = PointConv(mlp)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Input must be point cloud tensor of shape (n_points, n_features)  """
        pos = x[:, :3]
        print(x.shape)
        N, _ = x.shape  # (num_points, 3)
        pos = pos.view(N, -1)

        x = x[:, 3:]

        # get only x, y, z values per point

        # sample points (regions) by using iterative farthest point sampling (FPS)
        idx = fps(pos, ratio=self.sample_rate)

        # For each region, get all points which are within in the region (defined by radius r)
        row, col = radius(pos, pos[idx], self.radius)

        edge_index = torch.stack([col, row], dim=0)

        # Apply point net
        x1 = self.point_conv(x.float(), (pos.float(), pos[idx].float()), edge_index)

        print(x1.shape)
        return x


# class SetConvLayer(torch.nn.Module):
#     """
#     TODO
#     """
#     def __init__(self, sample_rate, radius):
#         super().__init__()
#         self.sample_rate = sample_rate
#         self.radius = radius
#         mlp = make_mlp(3, [64, 64, 128])
#         self.point_conv = PointConv(mlp)
#
#     def forward(self, x: torch.tensor) -> torch.tensor:
#         """ Input must be point cloud tensor of shape (n_points, n_features)  """
#         x = x[None, :, :3]
#         print(x.shape)
#         batch_size, N, _ = x.shape  # (batch_size, num_points, 3)
#         pos = x.view(batch_size * N, -1)
#         batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
#         for i in range(batch_size): batch[i] = i
#         batch = batch.view(-1)
#
#         data = None, pos, batch
#         x, pos, batch = data
#
#         # get only x, y, z values per point
#         # points = x[:, :3]
#         # sample points (regions) by using iterative farthest point sampling (FPS)
#         idx = fps(pos, batch, ratio=self.sample_rate)
#
#         # For each region, get all points which are within in the region (defined by radius r)
#         row, col = radius(pos, pos[idx], self.radius, batch, batch[idx])
#
#         edge_index = torch.stack([col, row], dim=0)
#
#         # Apply point net
#         x1 = self.point_conv(x, (pos.float(), pos[idx].float()), edge_index)
#
#         print(x1.shape)
#         return x
