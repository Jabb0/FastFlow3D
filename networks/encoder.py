import torch
import math


class PillarFeatureNet(torch.nn.Module):
    """
    Transform the raw point cloud data of shape (n_points, 3) into a representation of shape (n_points, 6).
    Each point consists of 6 features: (x_c, y_c, z_c, x_delta, y_delta, z_delta). TODO missing 2D of laser data.
    x_c, y_c, z_c: Center of the pillar to which the point belongs.
    x_delta, y_delta, z_delta: Offset from the pillar center to the point.

    References
    ----------
    .. [PointPillars] Alex H. Lang and Sourabh Vora and  Holger Caesar and Lubing Zhou and Jiong Yang and Oscar Beijbom
       PointPillars: Fast Encoders for Object Detection from Point Clouds
       https://arxiv.org/pdf/1812.05784.pdf
    """
    def __init__(self, n_pillars_x, n_pillars_y, in_features=8, out_features=64):
        super().__init__()
        self.n_pillars_x = n_pillars_x
        self.n_pillars_y = n_pillars_y

        self.in_features = in_features
        self.out_features = out_features

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.batch_norm = torch.nn.BatchNorm1d(out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, points, indices):
        """ Input must be the augmented point cloud of shape (batch_size, n_points, 6)
        and indices of shape (batch_size, n_points, 2)
        """

        # linear transformation
        embedded_points = self.linear(points)
        # FIXME
        # embedded_points = self.batch_norm(embedded_points)
        embedded_points = self.relu(embedded_points)

        indices = indices.long()

        # TODO replace for-loop over batches by torch operations (if possible)
        # TODO Check if this implementation is correct
        # Snap-To-Grid
        grid = torch.zeros(size=(embedded_points.shape[0], self.n_pillars_x, self.n_pillars_y, self.out_features))
        for i in range(embedded_points.shape[0]):
            batch_indices = indices[i]
            batch_x = embedded_points[i]

            n_pillars = torch.unique(batch_indices, dim=0).shape[0]  # number of different index pairs

            # add up all points, which have the same index
            batch_x = torch.zeros(n_pillars, 64, dtype=embedded_points.dtype).scatter_add_(0, batch_indices, batch_x)
            batch_indices = torch.unique(batch_indices, dim=0)  # get the corresponding indices

            # Snap-to-grid
            batch_grid = torch.zeros(size=(self.n_pillars_x, self.n_pillars_y, self.out_features))
            batch_grid[batch_indices[:, 0], batch_indices[:, 1], :] = batch_x

            grid[i] = batch_grid

        # Snap-to-grid
        # grid = torch.zeros(size=(self.n_pillars_x, self.n_pillars_y, self.out_features))
        # for i in range(x.shape[0]):
        #     x_idx, y_idx = indices[i].long()
        #     grid[x_idx, y_idx].add(x[i])

        return embedded_points, grid
