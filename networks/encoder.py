import torch
import math


class PillarFeatureNet(torch.nn.Module):
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

    def construct_sparse_grid_matrix(self, n_points, indices):
        # We now convert the grid_cell_indices into a grid cell lookup matrix
        # The matrix A has shape (N_points, grid_width * grid_height) and with Aij=1 if point i is in cell j
        # The cells are encoded as j = x * grid_width + y and thus give an unique encoding for each cell
        # E.g. if we have 512 cells in both directions and x=1, y=2 is encoded as 512 + 2 = 514.
        # Each new row of the grid (x-axis) starts at j % 512 = 0.
        # Init the matrix to only zeros

        # Construct the necessary indices. This is an array with [[first_dim0,....first_dimN],[second_dim0,...,second_dimN]]
        indices = torch.stack([
            indices,
            torch.arange(n_points, device=indices.device)
        ])

        grid_lookup_matrix = torch.sparse_coo_tensor(indices, torch.ones(n_points),
                                                     size=(self.n_pillars_x * self.n_pillars_y, n_points),
                                                     device=indices.device, requires_grad=False)
        # requires_grad=False ?
        return grid_lookup_matrix

    def forward(self, x, indices):
        # x -> [N, 64]
        # indices -> [N, 2], indicates which pillar belongs to each point
        """ Input must be the augmented point cloud of shape (n_points, 6) """
        # Calculate the mapping matrix from each point to its 1D encoded cell
        grid_lookup_matrix = self.construct_sparse_grid_matrix(x.shape[0], indices)
        # grid_lookup_matrix -> [512*512, N]
        # rows are points, columns are cells. 1 if the two are connected. This will sum up the points

        # We can now sum up the embeddings of all points as matrix multiplication
        # grid = [512*512, N] @ [N,64] -> [512*512, 64]
        grid = torch.mm(grid_lookup_matrix, x)
        # We now need to shape the 1D grid embedding into the actual 2D grid

        # Please refer to tests/test_encoder.py to check a test which check that this reformatting works
        grid3d = grid.view((self.n_pillars_x, self.n_pillars_y, x.shape[1]))
        grid3d = grid3d.permute((2, 0, 1))

        return grid3d


    def forward2(self, points, indices):
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