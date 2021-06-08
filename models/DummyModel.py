import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from argparse import ArgumentParser

from networks.TwoLayerNet import TwoLayerNet

# TODO: Add a general forward step and steps to compute the mean episode rewards at the end of each epoch


class DummyModel(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super(DummyModel, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams

        self.twolayer = TwoLayerNet(28 * 28, 10, self.hparams.hidden_dim)

    def forward(self, x):
        """
        The usual forward pass function of a torch module
        :param x:
        :return:
        """
        x = x.view(x.size(0), -1)
        x = self.twolayer(x)
        return x

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
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # Need to return the loss to do backprop
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Similar to the train step.
        Already has model.eval() and torch.nograd() set!
        :param batch:
        :param batch_idx:
        :return:
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # Log the metrics for this validation run. This directly logs to tensorboard
        # E.g. also compute accuracy here
        self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        """
        Similar to the train step.
        Already has model.eval() and torch.nograd() set!
        :param batch:
        :param batch_idx:
        :return:
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # Log the metrics for this validation run. This directly logs to tensorboard
        # E.g. also compute accuracy here
        self.log('test/loss', loss)

    def configure_optimizers(self):
        """
        Also pytorch lightning specific.
        Define the optimizers in here this will return the optimizer that is used to train this module.
        Also define learning rate scheduler in here. Not sure how this works...
        :return: The optimizer to use
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Method to add all command line arguments specific to this module.
        Used to dynamically add the correct arguments required.
        :param parent_parser: The current argparser to add the options to
        :return: the new argparser with the new options
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser