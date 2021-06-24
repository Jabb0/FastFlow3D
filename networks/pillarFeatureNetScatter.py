import torch


class PillarFeatureNetScatter(torch.nn.Module):
    """
    Transform the raw point cloud data of shape (n_points, 3) into a representation of shape (n_points, 6).
    Each point consists of 6 features: (x_c, y_c, z_c, x_delta, y_delta, z_delta, laser_feature1, laser_feature2).
    x_c, y_c, z_c: Center of the pillar to which the point belongs.
    x_delta, y_delta, z_delta: Offset from the pillar center to the point.

    References
    ----------
    .. [PointPillars] Alex H. Lang and Sourabh Vora and  Holger Caesar and Lubing Zhou and Jiong Yang and Oscar Beijbom
       PointPillars: Fast Encoders for Object Detection from Point Clouds
       https://arxiv.org/pdf/1812.05784.pdf
    """
    def __init__(self, n_pillars_x, n_pillars_y, out_features=64):
        super().__init__()
        self.n_pillars_x = n_pillars_x
        self.n_pillars_y = n_pillars_y

        self.out_features = out_features

    def forward(self, x, indices):
        # pc input is (batch_size, N_points, 64) with 64 being the embedding dimension of each point
        # in indices we have (batch_size, N_points, 2) with the 2D coordinates in the grid
        # We want to scatter into a n_pillars_x and n_pillars_y grid
        # Thus we should allocate a tensor of the desired shape (batch_size, n_pillars_x, n_pillars_y, 64)
        # To make things easier we transform the 2D indices into 1D indices
        # The cells are encoded as j = x * grid_width + y and thus give an unique encoding for each cell
        # E.g. if we have 512 cells in both directions and x=1, y=2 is encoded as 512 + 2 = 514.
        # Each new row of the grid (x-axis) starts at j % 512 = 0.
        # Init the matrix to only zeros
        indices = indices[:, :, 0] * self.n_pillars_x + indices[:, :, 1]
        # The indices also need to have a 3D dimension. That dimension is the feature dimension of the inputs.
        # We need to repeat the grid cell index such that all feature dimensions are summed up.
        # Note: we use .expand here. Expand does not actually create new memory.
        #   It just views the same entry multiple times.
        #   But as the index does not need a grad nor is changed in place this is fine.
        # First unsqueeze the (batch, points) vector to (batch, points, 1)
        # Then expand the 1 dimension such that the index is defined for all the features desired
        # -1 means do not change this dim. (batch, points,
        indices = indices.unsqueeze(-1).expand(-1, -1, x.size(2))
        # Now indices is (batch_size, N_points) now
        # Construct the desired tensor
        grid = torch.zeros((x.size(0), self.n_pillars_x * self.n_pillars_y, x.size(2)), device=x.device)
        # And now perform the infamous scatter_add_ that changes the grid in place
        # the source (x) and the indices matrix are now 2 dimensional with (batch_size, points)
        # The batch dimension stays the same. But the cells are looked up using the index
        # thus: grid[batch][index[batch][point]] += x[batch][point]
        grid.scatter_add_(1, indices, x)
        # Do a test if actually multiple
        # the later 2D convolutions want (batch, channels, H, W) of the image
        # thus make the grid (batch, channels, cells)
        grid = grid.permute((0, 2, 1))
        # and make the cells 2D again
        grid = grid.unflatten(2, (self.n_pillars_x, self.n_pillars_y))

        return grid
