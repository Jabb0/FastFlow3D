import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from pytorch_lightning.metrics import functional as FM
from argparse import ArgumentParser

from networks import PillarFeatureNet, ConvEncoder


class FastFlow3DModel(pl.LightningModule):
    def __init__(self, x_max, x_min, y_max, y_min, grid_cell_size, point_features=6,
                 learning_rate=1e-6,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999):
        super(FastFlow3DModel, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams

        self._pillar_feature_net = PillarFeatureNet(x_max, x_min, y_max, y_min, grid_cell_size,
                                                    in_features=point_features, out_features=64)
        # Note: There is also xavier_normal_ but the paper does not state which one they used.
        torch.nn.init.xavier_uniform_(self._pillar_feature_net.weight)

        self._conv_encoder_net = ConvEncoder(in_channels=64, out_channels=256)
        torch.nn.init.xavier_uniform_(self._conv_encoder_net.weight)
        # TODO: Remaining networks


    def forward(self, x):
        """
        The usual forward pass function of a torch module
        :param x: (batch_size, (t-1, t0)) a batch of the two point cloud images
        :return:
        """
        # (batch_size, 2, N_points, N_features)
        point_cloud_prev, point_cloud_cur = x

        # 1. Do scene encoding of each point cloud to get the grid with pillar embeddings
        # Input is a point cloud each with shape (N_points, point_features)
        # TODO: Need to add the indices
        pillar_embeddings_prev = self._pillar_feature_net(point_cloud_prev)
        pillar_embeddings_cur = self._pillar_feature_net(point_cloud_cur)
        # Output is a 512x512x64 2D embedding representing the point clouds

        # 2. Apply the U-net encoder step
        # Note that weight sharing is used here. The same U-net is used for both point clouds.
        # Corresponds to F, L and R from the paper
        prev_64_conv, prev_128_conv, prev_256_conv = self._conv_encoder_net(pillar_embeddings_prev)
        cur_64_conv, cur_128_conv, cur_256_conv = self._conv_encoder_net(pillar_embeddings_cur)

        # 3. Apply the U-net decoder with skip connections
        # TODO:

        # 4. Apply the unpillar operation
        # TODO:

        # Return the final motion prediction of size (N_points_cur, 3)
        return x

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
        loss = F.mse_loss(y_hat, y)
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
