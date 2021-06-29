import torch


class UnpillarNetwork(torch.nn.Module):

    def __init__(self, n_pillars_x, n_pillars_y):
        super().__init__()
        self.n_pillars_x = n_pillars_x
        self.n_pillars_y = n_pillars_y

        self._Y = torch.nn.Linear(128, 32)
        self._Z = torch.nn.Linear(32, 3)

    def construct_sparse_grid_matrix(self, n_points, indices):
        # We now convert the grid_cell_indices into a grid cell lookup matrix
        # The matrix A has shape (N_points, grid_width * grid_height) and with Aij=1 if point i is in cell j
        # The cells are encoded as j = x * grid_width + y and thus give an unique encoding for each cell
        # E.g. if we have 512 cells in both directions and x=1, y=2 is encoded as 512 + 2 = 514.
        # Each new row of the grid (x-axis) starts at j % 512 = 0.

        # Note that this matrix is the inverse of the sparse matrix used for pillarization

        # Construct the necessary indices.
        indices = torch.stack([
            torch.arange(n_points, device=indices.device),
            indices,
        ])

        grid_lookup_matrix = torch.sparse_coo_tensor(indices, torch.ones(n_points),
                                                     size=(n_points, self.n_pillars_x * self.n_pillars_y),
                                                     device=indices.device, requires_grad=False)
        # requires_grad=False ?
        return grid_lookup_matrix

    def forward(self, grid_flow_embeddings, point_cloud, grid_indices):
        """

        :param grid_flow_embeddings:
        :param point_cloud:
        :param grid_indices:
        :return:
        """
        # print(f"Number of points in the point cloud {point_cloud.shape}")
        grid_lookup_matrix = self.construct_sparse_grid_matrix(point_cloud.shape[0], grid_indices)
        # Perform lookup of the correct flow embeddings for each point
        # This is the ungrid operation
        # Output is a (N_points, 64) matrix where each point is assigned its cell embedding
        # But the input needs to be again in the 1D cell encoding of the grid_lookup matrix

        # grid_flow_embeddings -> [64, 512, 512]
        # Please refer to tests/test_encoder.py to check a test which check that this reformatting work
        grid_flow_embeddings = grid_flow_embeddings.reshape((grid_flow_embeddings.shape[0],
                                                             self.n_pillars_x * self.n_pillars_y))
        grid_flow_embeddings = grid_flow_embeddings.permute((1, 0))

        # print(f"grid_lookup_matrix {grid_lookup_matrix.size()}")
        point_embeddings = torch.sparse.mm(grid_lookup_matrix, grid_flow_embeddings)
        # print(f"point_embeddings {point_embeddings.size()}")
        # print(f"point_cloud {point_cloud.size()}")
        # Concatenate the cell embeddings with the point embeddings for each point
        full_embeddings = torch.cat([point_embeddings, point_cloud], dim=1)
        # print(f"full_embeddings {full_embeddings.size()}")
        # Output is a (N_point, 128) embedding matrix
        # Infer the flow for each point
        full_embeddings = self._Y(full_embeddings)
        full_embeddings = self._Z(full_embeddings)

        return full_embeddings
