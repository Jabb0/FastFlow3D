import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from pytorch_lightning.metrics import functional as FM
from argparse import ArgumentParser

from networks import PillarFeatureNetScatter, ConvEncoder, ConvDecoder, UnpillarNetworkScatter, PointFeatureNet
from .utils import init_weights, augment_index


class FastFlow3DModelScatter(pl.LightningModule):
    def __init__(self, n_pillars_x, n_pillars_y,
                 point_features=8,
                 learning_rate=1e-6,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999):
        super(FastFlow3DModelScatter, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams

        self._point_feature_net = PointFeatureNet(in_features=point_features, out_features=64)

        self._pillar_feature_net = PillarFeatureNetScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y,
                                                           out_features=64)
        self._pillar_feature_net.apply(init_weights)

        self._conv_encoder_net = ConvEncoder(in_channels=64, out_channels=256)
        self._conv_encoder_net.apply(init_weights)

        self._conv_decoder_net = ConvDecoder()
        self._conv_decoder_net.apply(init_weights)

        self._unpillar_network = UnpillarNetworkScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y)
        self._unpillar_network.apply(init_weights)

        self._n_pillars_x = n_pillars_x

    def _transform_point_cloud_to_embeddings(self, pc, mask):
        # Flatten the first two dimensions to get the points as batch dimension
        previous_batch_pc_embedding = self._point_feature_net(pc.flatten(0, 1))
        # Output is (batch_size * points, embedding_features)
        # Set the points to 0 that are just there for padding
        # TODO: One could ignore the mask and only give indices for the actually defined entries.
        #  However it is unclear if then backprop is implemented
        #  as the documentation states it is only defined for src.shape == index.shape
        #  for scatter_add_ that is used in the pillarization part
        #  however, then batching of the indices does not work anymore. Therefore this is no real possibility.
        previous_batch_pc_embedding[mask.flatten(0, 1), :] = 0
        # Retransform into batch dimension (batch_size, max_points, embedding_features)
        previous_batch_pc_embedding = previous_batch_pc_embedding.unflatten(0,
                                                                            (pc.size(0),
                                                                             pc.size(1)))
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
        # batch_pc = (batch_size, N, 8) | batch_grid = (n_batch, N, 2) | batch_mask = (n_batch, N)

        # Pass the whole batch of point clouds to get the embedding for each point in the cloud
        # Input pc is (batch_size, max_n_points, features_in)
        # per each point, there are 8 features: [cx, cy, cz,  Δx, Δy, Δz, l0, l1], as stated in the paper
        previous_batch_pc_embedding = self._transform_point_cloud_to_embeddings(previous_batch_pc.float(),
                                                                                previous_batch_mask)
        # previous_batch_pc_embedding = [n_batch, N, 64]
        # Output pc is (batch_size, max_n_points, embedding_features)
        current_batch_pc_embedding = self._transform_point_cloud_to_embeddings(current_batch_pc.float(),
                                                                               current_batch_mask)

        # To make things easier we transform the 2D indices into 1D indices
        # The cells are encoded as j = x * grid_width + y and thus give an unique encoding for each cell
        # E.g. if we have 512 cells in both directions and x=1, y=2 is encoded as 512 + 2 = 514.
        # Each new row of the grid (x-axis) starts at j % 512 = 0.
        previous_batch_grid = augment_index(previous_batch_grid, previous_batch_pc_embedding.size(2), self._n_pillars_x)
        current_batch_grid = augment_index(current_batch_grid, current_batch_pc_embedding.size(2), self._n_pillars_x)

        # Now we need to scatter the points into their 2D matrix
        # batch_pc_embeddings -> (batch_size, N, 64)
        # batch_grid -> (batch_size, N, 64)
        previous_pillar_embeddings = self._pillar_feature_net(previous_batch_pc_embedding, previous_batch_grid)
        current_pillar_embeddings = self._pillar_feature_net(current_batch_pc_embedding, current_batch_grid)
        # pillar_embeddings = (batch_size, 64, 512, 512)

        # 2. Apply the U-net encoder step
        # Note that weight sharing is used here. The same U-net is used for both point clouds.
        # Corresponds to F, L and R from the paper
        prev_64_conv, prev_128_conv, prev_256_conv = self._conv_encoder_net(previous_pillar_embeddings)
        cur_64_conv, cur_128_conv, cur_256_conv = self._conv_encoder_net(current_pillar_embeddings)

        # 3. Apply the U-net decoder with skip connections
        grid_flow_embeddings = self._conv_decoder_net(previous_pillar_embeddings, prev_64_conv,
                                                      prev_128_conv, prev_256_conv,
                                                      current_pillar_embeddings, cur_64_conv,
                                                      cur_128_conv, cur_256_conv)

        # grid_flow_embeddings -> [batch_size, 64, 512, 512]

        # 4. Apply the unpillar and flow prediction operation
        predictions = self._unpillar_network(grid_flow_embeddings, current_batch_pc_embedding, current_batch_grid)

        # List of batch size many output with each being a (N_points, 3) flow prediction.
        return predictions

    def masked_mse_loss(self, input, target, mask):
        return (mask * (input - target) ** 2).mean()

    def general_step(self, batch, batch_idx, mode):
        """
        A function to share code between all different steps.
        :param batch: the batch to perform on
        :param batch_idx: the id of the batch
        :param mode: str of "train", "val", "test". Useful if specific things are required.
        :return:
        """
        x, y = batch
        y_hat = self(x)
        # x is a list of input batches with the necessary data
        # For loss calculation we need to know which elements are actually present and not padding
        # Therefore we need the mast of the current frame as batch tensor
        # It is True for all points that just are padded and of size (batch_size, max_points)
        # Invert the matrix for our purpose
        current_frame_mask = ~x[1][2].unsqueeze(-1)

        # The first 3 dimensions are the actual flow. The last dimension is the class id.
        y_flow = y[:, :, :3]
        # TODO: Is the masking properly implemented?!
        # TODO: Add weights for the background class
        loss = self.masked_mse_loss(y_hat, y_flow, current_frame_mask)
        return loss

    def training_step(self, batch, batch_idx):
        """
        This method is specific to pytorch lightning.
        It is called for each minibatch that the model should be trained for.
        Basically a part of the normal training loop is just moved here.

        model.train() is already set!
        :param batch: (data, target) of batch size
        :param batch_idx: the id of this batch e.g. for discounting?
        :return:
        """
        loss = self.general_step(batch, batch_idx, "train")
        # Automatically reduces this metric after each epoch
        # Note: There is also a log_dict function that can log multiple metrics at a time.
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Return loss for backpropagation
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Similar to the train step.
        Already has model.eval() and torch.nograd() set!
        :param batch:
        :param batch_idx:
        :return:
        """
        loss = self.general_step(batch, batch_idx, "val")
        # Automatically reduces this metric after each epoch
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """
        Similar to the train step.
        Already has model.eval() and torch.nograd() set!
        :param batch:
        :param batch_idx:
        :return:
        """
        loss = self.general_step(batch, batch_idx, "test")
        # Automatically reduces this metric after each epoch
        self.log('test/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Also pytorch lightning specific.
        Define the optimizers in here this will return the optimizer that is used to train this module.
        Also define learning rate scheduler in here. Not sure how this works...
        :return: The optimizer to use
        """
        # Defaults are the same as for pytorch
        betas = (
            self.hparams.adam_beta_1 if self.hparams.adam_beta_1 is not None else 0.9,
            self.hparams.adam_beta_1 if self.hparams.adam_beta_2 is not None else 0.999)

        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=betas, weight_decay=0)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Method to add all command line arguments specific to this module.
        Used to dynamically add the correct arguments required.
        :param parent_parser: The current argparser to add the options to
        :return: the new argparser with the new options
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-6)
        return parser
