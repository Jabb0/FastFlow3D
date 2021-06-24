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

        # Init the matrix to only zeros
        # Now indices is (batch_size, N_points, features)
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
