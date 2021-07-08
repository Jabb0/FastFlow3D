import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from collections import defaultdict
from utils import str2bool


class BaseModel(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-6,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999,
                 background_weight=0.1):
        super(BaseModel, self).__init__()
        self._background_weight = background_weight
        # ----- Metrics information -----
        # TODO delete no flow class
        self._classes = [(0, 'background'), (1, 'vehicle'), (2, 'pedestrian'), (3, 'sign'), (4, 'cyclist')]
        self._thresholds = [(1, '1_1'), (0.1, '1_10')]  # 1/1 = 1, 1/10 = 0.1
        self._min_velocity = 0.5  # If velocity higher than 0.5 then it is considered as the object is moving

    def compute_metrics(self, y, y_hat, labels):
        """

        :param y: predicted data (points, 3)
        :param y_hat: ground truth (points, 3)
        :param labels: class labels for each point (points, 3)
        :return:
        """
        squared_root_difference = torch.sqrt(torch.sum((y - y_hat) ** 2, dim=1))
        # We compute the weighting vector for background_points
        # weights is a mask which background_weight value for backgrounds and 1 for no backgrounds, in order to
        # downweight the background points
        weights = torch.ones((squared_root_difference.shape[0]),
                             device=squared_root_difference.device,
                             dtype=squared_root_difference.dtype)  # weights -> (batch_size * N)
        weights[labels == 0] = self._background_weight

        loss = torch.sum(weights * squared_root_difference) / torch.sum(weights)
        # ---------------- Computing rest of metrics (Paper Table 3)-----------------

        # We compute a dictionary with 3 different metrics:
        # mean: L2 mean. This computes the L2 mean per each class. We also distinguish the state of a class element,
        # which can be moving or stationary. A class element is considered as it is moving when the flow vector magnitude
        # is >= self._min_velocity and it is stationary when is less than _min_velocity.
        # Then, metrics = {mean: L2_mean,
        #                   ...: .....}
        # {mean: {all: {label1: xxx, label2: yyy, ...}, moving: {label1: xxx, label2: yyy}, stationary: {...}}, otherMetric: {...}, ...}
        # "all" does not distinghish the label state

        # We also compute the accuracy, which stands for the percentage of points with error below 0.1 m/s and 1.0 m/s (self.self._thresholds)
        # Depending on the threshold, we will have an item in the dictionary which is:
        # {mean: {...}, 1_1: {all: {label1: xxx, label2: yyy, ...}, moving: {label1: xxx, label2: yyy}, stationary: {...}}}
        # 1_1 stands for 1 / 1, which is 1 m/s, 1_10 stands for 1/10, which is 0.1 m/s.

        # Use detach as those metrics do not need a gradient
        L2_without_weighting = squared_root_difference.detach()
        flow_vector_magnitude = torch.sqrt(torch.sum(y_hat.detach() ** 2, dim=1))

        L2_mean = {}
        nested_dict = lambda: defaultdict(nested_dict)
        L2_thresholds = nested_dict()  # To create nested dict
        all_labels = {}  # L2 mean for labels without distinguish state
        moving_labels = {}  # L2 mean computed for each of the labels but only taking into account moving points
        stationary_labels = {}  # L2 mean computed for each of the labels but only taking into account stationary points
        for label, class_name in self._classes:
            # ----------- Computing L2 mean -------------

            # --- stationary, moving and all (sum of both) elements of the class ---
            # To generate boolean mask that will help us filter elements of the label we are iterating
            label_mask = labels == label

            # with label_mask we only take items of label we are iterating
            L2_label = L2_without_weighting[label_mask]
            flow_vector_magnitude_label = flow_vector_magnitude[label_mask]

            stationary = L2_label[flow_vector_magnitude_label < self._min_velocity]  # Extract stationary flows
            moving = L2_label[flow_vector_magnitude_label >= self._min_velocity]  # Extract flows in movement

            if L2_label.numel() != 0:
                mean_label_all = L2_label.mean()
                all_labels[class_name] = mean_label_all
            if moving.numel() != 0:
                mean_label_moving = moving.mean()
                moving_labels[class_name] = mean_label_moving
            if stationary.numel() != 0:
                mean_label_stationary = stationary.mean()
                stationary_labels[class_name] = mean_label_stationary

            # ----------- Computing L2 accuracy with threshold -------------
            for threshold, name in self._thresholds:
                if L2_label.numel() != 0:
                    all_accuracy = (L2_label <= threshold).float().mean()
                    L2_thresholds[name]['all'][class_name] = all_accuracy
                if stationary.numel() != 0:
                    stationary_accuracy = (stationary <= threshold).float().mean()
                    L2_thresholds[name]['stationary'][class_name] = stationary_accuracy
                if moving.numel() != 0:
                    moving_accuracy = (moving <= threshold).float().mean()
                    L2_thresholds[name]['moving'][class_name] = moving_accuracy

        L2_mean['all'] = all_labels
        L2_mean['moving'] = moving_labels
        L2_mean['stationary'] = stationary_labels

        metrics = {'mean': L2_mean}
        metrics.update(L2_thresholds)
        return loss, metrics
        #return loss, {}

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
        # It is True for all points that just are NOT padded and of size (batch_size, max_points)
        current_frame_masks = x[1][2]

        # Remove all points that are padded
        y = y[current_frame_masks]
        y_hat = y_hat[current_frame_masks]
        # This will yield a (n_real_points, 3) tensor with the batch size being included already

        # The first 3 dimensions are the actual flow. The last dimension is the class id.
        y_flow = y[:, :3]
        # Loss computation
        labels = y[:, -1].int()  # Labels are actually integers so lets convert them
        # Remove datapoints with no flow assigned (class -1)
        mask = labels != -1
        y_hat = y_hat[mask]
        y_flow = y_flow[mask]
        labels = labels[mask]

        loss, metrics = self.compute_metrics(y_flow, y_hat, labels)

        return loss, metrics

    def log_metrics(self, loss, metrics, phase):
        # phase should be training, validation or test
        metrics_dict = {}
        for metric in metrics:
            for state in metrics[metric]:
                for label in metrics[metric][state]:
                    metrics_dict[f'{phase}/{metric}/{state}/{label}'] = metrics[metric][state][label]

        # Do not log the in depth metrics in the progress bar
        self.log(f'{phase}/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(metrics_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)

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
        print(f"training step finished, loss {loss}")
        print(f"training step finished, metrics {metrics}")
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
        parser.add_argument('--use_group_norm', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--background_weight', default=0.1, type=float)
        return parser
