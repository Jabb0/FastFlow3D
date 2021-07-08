import torch

from networks import PillarFeatureNetScatter, ConvEncoder, ConvDecoder, UnpillarNetworkScatter, PointFeatureNet
from .utils import init_weights
from models.BaseModel import BaseModel


class FastFlow3DModelScatter(BaseModel):
    def __init__(self, n_pillars_x, n_pillars_y,
                 background_weight=0.1,
                 point_features=8,
                 use_group_norm=False,
                 learning_rate=1e-6,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999):
        super(FastFlow3DModelScatter, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams

        self._point_feature_net = PointFeatureNet(in_features=point_features, out_features=64)
        self._point_feature_net.apply(init_weights)

        self._pillar_feature_net = PillarFeatureNetScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y,
                                                           out_features=64)
        self._pillar_feature_net.apply(init_weights)

        self._conv_encoder_net = ConvEncoder(in_channels=64, out_channels=256, use_group_norm=use_group_norm)
        self._conv_encoder_net.apply(init_weights)

        self._conv_decoder_net = ConvDecoder()
        self._conv_decoder_net.apply(init_weights)

        self._unpillar_network = UnpillarNetworkScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y)
        self._unpillar_network.apply(init_weights)

        self._n_pillars_x = n_pillars_x
        self._background_weight = background_weight

        # ----- Metrics information -----
        # TODO delete no flow class
        self._classes = [(0, 'background'), (1, 'vehicle'), (2, 'pedestrian'), (3, 'sign'), (4, 'cyclist')]
        self._thresholds = [(1, '1_1'), (0.1, '1_10')]  # 1/1 = 1, 1/10 = 0.1
        self._min_velocity = 0.5  # If velocity higher than 0.5 then it is considered as the object is moving

    def _transform_point_cloud_to_embeddings(self, pc, mask):
        pc_flattened = pc.flatten(0, 1)
        mask_flattened = mask.flatten(0, 1)
        # Init the result tensor for our data. This is necessary because the point net
        # has a batch norm and this needs to ignore the masked points
        previous_batch_pc_embedding = torch.zeros((pc_flattened.size(0), 64),
                                                  device=pc.device, dtype=pc.dtype)
        # Flatten the first two dimensions to get the points as batch dimension
        previous_batch_pc_embedding[mask_flattened] = self._point_feature_net(pc_flattened[mask_flattened])
        # This allows backprop towards the MLP: Checked with backward hooks. Gradient is present.
        # Output is (batch_size * points, embedding_features)
        # Retransform into batch dimension (batch_size, max_points, embedding_features)
        previous_batch_pc_embedding = previous_batch_pc_embedding.unflatten(0, (pc.size(0), pc.size(1)))
        # 241.307 MiB    234
        return previous_batch_pc_embedding

    def forward(self, x):
        """
        The usual forward pass function of a torch module
        :param x:
        :return:
        """
        # 1. Do scene encoding of each point cloud to get the grid with pillar embeddings
        # Input is a point cloud each with shape (N_points, point_features)

        # The input here is more complex as we deal with a batch of point clouds
        # that do not have a fixed amount of points
        # x is a tuple of two lists representing the batches
        previous_batch, current_batch = x
        previous_batch_pc, previous_batch_grid, previous_batch_mask = previous_batch
        current_batch_pc, current_batch_grid, current_batch_mask = current_batch
        # For some reason the datatype of the input is not changed to correct precision
        previous_batch_pc = previous_batch_pc.type(self.dtype)
        current_batch_pc = current_batch_pc.type(self.dtype)

        # batch_pc = (batch_size, N, 8) | batch_grid = (n_batch, N, 2) | batch_mask = (n_batch, N)
        # The grid indices are (batch_size, max_points) long. But we need them as
        # (batch_size, max_points, feature_dims) to work. Features are in all necessary cases 64.
        # Expand does only create multiple views on the same datapoint and not allocate extra memory
        current_batch_grid = current_batch_grid.unsqueeze(-1).expand(-1, -1, 64)
        previous_batch_grid = previous_batch_grid .unsqueeze(-1).expand(-1, -1, 64)

        # Pass the whole batch of point clouds to get the embedding for each point in the cloud
        # Input pc is (batch_size, max_n_points, features_in)
        # per each point, there are 8 features: [cx, cy, cz,  Δx, Δy, Δz, l0, l1], as stated in the paper
        previous_batch_pc_embedding = self._transform_point_cloud_to_embeddings(previous_batch_pc,
                                                                                previous_batch_mask)
        # previous_batch_pc_embedding = [n_batch, N, 64]
        # Output pc is (batch_size, max_n_points, embedding_features)
        current_batch_pc_embedding = self._transform_point_cloud_to_embeddings(current_batch_pc,
                                                                               current_batch_mask)

        # Now we need to scatter the points into their 2D matrix
        # batch_pc_embeddings -> (batch_size, N, 64)
        # batch_grid -> (batch_size, N, 64)
        previous_pillar_embeddings = self._pillar_feature_net(previous_batch_pc_embedding, previous_batch_grid)
        current_pillar_embeddings = self._pillar_feature_net(current_batch_pc_embedding, current_batch_grid)
        # pillar_embeddings = (batch_size, 64, 512, 512)

        # Concatenate the previous and current batches along a new dimension.
        # This allows to have twice the amount of entries in the forward pass
        # of the encoder which is good for batch norm.
        pillar_embeddings = torch.stack((previous_pillar_embeddings, current_pillar_embeddings), dim=1)
        # This is now (batch_size, 2, 64, 512, 512) large
        pillar_embeddings = pillar_embeddings.flatten(0, 1)
        # Flatten into (batch_size * 2, 64, 512, 512) for encoder forward pass.

        # 2. Apply the U-net encoder step
        # Note that weight sharing is used here. The same U-net is used for both point clouds.
        # Corresponds to F, L and R from the paper. Number corresponds to the depth.
        conv_64, conv_128, conv_256 = self._conv_encoder_net(pillar_embeddings)
        # Output is (batch_size * 2, C, W, H) for each layer.
        # Now unflatten the first dimension again such that they are viewed as
        # (batch_size, 2, C, W, H) with the second dimension being (prev, cur)
        # This removed the need to concatenate the two layers on as we can just view their channels together
        batch_size = current_pillar_embeddings.size(0)
        pillar_embeddings = pillar_embeddings.unflatten(0, (batch_size, 2))
        conv_64 = conv_64.unflatten(0, (batch_size, 2))
        conv_128 = conv_128.unflatten(0, (batch_size, 2))
        conv_256 = conv_256.unflatten(0, (batch_size, 2))

        # 3. Apply the U-net decoder with skip connections
        grid_flow_embeddings = self._conv_decoder_net(pillar_embeddings, conv_64, conv_128, conv_256)

        # grid_flow_embeddings -> [batch_size, 64, 512, 512]

        # 4. Apply the unpillar and flow prediction operation
        predictions = self._unpillar_network(grid_flow_embeddings, current_batch_pc_embedding, current_batch_grid)

        # List of batch size many output with each being a (N_points, 3) flow prediction.
        return predictions
