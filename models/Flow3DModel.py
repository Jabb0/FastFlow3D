import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from argparse import ArgumentParser

from networks.flownet3d.pointFeatureNet import PointFeatureNet
from networks.flownet3d.pointMixture import PointMixtureNet
from networks.flownet3d.flowRefinement import FlowRefinementNet


class Flow3DModel(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-6,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999):
        super(Flow3DModel, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams

        self._point_feature_net = PointFeatureNet(in_channels=8)  # TODO Change to 5
        self._point_mixture = PointMixtureNet()
        self._flow_refinement = FlowRefinementNet()
        self._final_linear = torch.nn.Linear(in_features=128, out_features=3)

    def forward(self, x):
        """
        The usual forward pass function of a torch module
        :param x:
        :return:
        """
        previous_batch, current_batch = x
        previous_batch_pc, _, previous_batch_mask = previous_batch
        current_batch_pc, _, current_batch_mask = current_batch

        #previous_batch_pc = torch.randint(low=0, high=100, size=(2, 10000, 8)).float()
        print(previous_batch_pc.shape)

        # TODO We need raw point cloud, i.e. x is of shape (n_points, 5), where the first 3 dims corresponds to x, y, z and the last two are the laser features
        # TODO Pass both point clouds through PointMixtureNet, FlowRefinementNet, LinearLayer
        previous_batch_pc = self._point_feature_net(previous_batch_pc.float())
        current_batch_pc = self._point_feature_net(current_batch_pc.float())

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
        batch_y_hat = self(x)
        # y_hat is a list of point clouds because they do not have the same shape
        # We need to compute the loss for each point cloud and return the mean over them
        total_loss = torch.zeros(1, device=self.device)
        total_points = 0
        for i, y_hat in enumerate(batch_y_hat):
            # Get the x,y,z flow targets and the label
            flow_target = y[i][:, :3]
            # label = y[i][:, 3]

            # print(f"y shape {y[i].shape}")
            # print(f"y_hat shape {y_hat.shape}")
            # TODO: This does not take a weighting of classes into account as described into the paper
            loss = F.mse_loss(y_hat, flow_target.float().to(y_hat.device))
            points = y_hat.shape[0]
            total_loss += points * loss
            total_points += points
        return total_loss / total_points

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
