import torch


class UnpillarNetworkScatter(torch.nn.Module):

    def __init__(self, n_pillars_x, n_pillars_y):
        super().__init__()
        self.n_pillars_x = n_pillars_x
        self.n_pillars_y = n_pillars_y

        self._Y = torch.nn.Linear(128, 32)
        self._Z = torch.nn.Linear(32, 3)

    def forward(self, grid_flow_embeddings, point_cloud, indices):
        """

        :param grid_flow_embeddings: (batch_size, 64, n_pillars_x, n_pillars_y) 2D image of flow predictions
        :param point_cloud: (batch_size, max_points, 64) point embeddings.
        :param indices: (batch_size, max_points, 64) lookup matrix of the grid indices
        :return:
        """
        # Lookup the grid cell embeddings
        # Output will be (batch_size, max_points, 64) where each
        # First encode the indices again
        # print("indices shape", indices.size())
        # Indices is now (batch_size, max_points)
        # Permute the grid into (batch_size, n_x, n_y, 64) again
        grid_flow_embeddings = grid_flow_embeddings.permute((0, 2, 3, 1))
        # Flatten the dimensions again such that it can be indexed using the 1d encoded index
        grid_flow_embeddings = grid_flow_embeddings.flatten(1, 2)
        # print("flow embeddings shape", grid_flow_embeddings.size())

        cell_embeddings = grid_flow_embeddings.gather(dim=1, index=indices.long())
        # print("cell embeddings shape", cell_embeddings.size())
        # Concatenate the cell embeddings with the point embeddings for each point
        full_embeddings = torch.cat([cell_embeddings, point_cloud], dim=2)
        # print(f"full_embeddings {full_embeddings.size()}")
        # Output is a (N_point, 128) embedding matrix
        # Infer the flow for each point
        # TODO: Maybe flatten this again? Not strictly necessary for MLP pass only
        full_embeddings = self._Y(full_embeddings)
        full_embeddings = self._Z(full_embeddings)

        return full_embeddings
