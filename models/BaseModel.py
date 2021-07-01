import pytorch_lightning as pl
import torch
from argparse import ArgumentParser


class BaseModel(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-6,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999):
        super(BaseModel, self).__init__()
        # ----- Metrics information -----
        # TODO delete no flow class
        self._classes = [(0, 'background'), (1, 'vehicle'), (2, 'pedestrian'), (3, 'sign'), (4, 'cyclist')]
        self._thresholds = [(1, '1_1'), (0.1, '1_10')]  # 1/1 = 1, 1/10 = 0.1

    def compute_metrics(self, y, y_hat, mask, labels, background_weight):
        # y, y_hat = (batch_size, N, 3)
        # mask = (batch_size, N)
        # weights = (batch_size, N, 1)
        # labels = (batch_size, N)
        # background_weight = float

        # First we flatten the batch dimension since it will make computations easier
        # For computing the metrics it is not needed to distinguish between batches
        y = y.flatten(0, 1)
        y_hat = y_hat.flatten(0, 1)
        mask = mask.flatten(0, 2)
        labels = labels.flatten(0, 1)
        # Flattened versions -> Second dimension is batch_size * N

        squared_root_difference = torch.sqrt(torch.sum((y - y_hat) ** 2, dim=1))
        # We mask the padding points
        squared_root_difference = squared_root_difference[mask]
        # We compute the weighting vector for background_points
        # weights is a mask which background_weight value for backgrounds and 1 for no backgrounds, in order to
        # downweight the background points
        weights = torch.ones((mask.shape[0]))  # weights -> (batch_size * N)
        weights[mask == False] = 0
        weights[labels == -1] = 0
        weights[labels == 0] = background_weight

        loss = torch.sum(weights * squared_root_difference) / torch.sum(weights)
        # ---------------- Computing rest of metrics -----------------
        metrics = {}
        # --- Computing L2 with threshold ---
        for threshold, name in self._thresholds:
            L2_list = {}
            for label, class_name in self._classes:
                correct_prediction = (squared_root_difference < threshold).sum()
                metric = correct_prediction / squared_root_difference.shape[0]
                L2_list[class_name] = metric
            metrics[name] = L2_list
        return loss, metrics

    # def compute_metrics(self, y, y_hat, mask, label):
    #    # L2 with threshold 1 m/s
    #    (mask * (y - y_hat) ** 2))
    # https://stackoverflow.com/questions/53906380/average-calculation-in-python
    # https://www.geeksforgeeks.org/numpy-maskedarray-mean-function-python/

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
        # Loss computation
        labels = y[:, :, -1]
        loss, metrics = self.compute_metrics(y_flow, y_hat, current_frame_mask, labels, self._background_weight)

        return loss, metrics

    def log_metrics(self, loss, metrics, phase):
        # phase should be training, validation or test
        metrics_dict = {}
        metrics_dict[f'{phase}/loss'] = loss
        for metric in metrics:
            for label in metrics[metric]:
                metrics_dict[f'{phase}/{metric}/{label}'] = metrics[metric][label]

        self.log_dict(metrics_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
        phase = "train"
        loss, metrics = self.general_step(batch, batch_idx, phase)
        # Automatically reduces this metric after each epoch
        self.log_metrics(loss, metrics, phase)
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
        phase = "val"
        loss, metrics = self.general_step(batch, batch_idx, phase)
        # Automatically reduces this metric after each epoch
        self.log_metrics(loss, metrics, phase)

    def test_step(self, batch, batch_idx):
        """
        Similar to the train step.
        Already has model.eval() and torch.nograd() set!
        :param batch:
        :param batch_idx:
        :return:
        """
        phase = "test"
        loss, metrics = self.general_step(batch, batch_idx, phase)
        # Automatically reduces this metric after each epoch
        self.log_metrics(loss, metrics, phase)

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
