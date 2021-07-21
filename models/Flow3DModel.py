from argparse import ArgumentParser

import torch

from models.BaseModel import BaseModel
from networks.flownet3d.flowRefinement import FlowRefinementNet
from networks.flownet3d.pointFeatureNet import PointFeatureNet
from networks.flownet3d.pointMixture import PointMixtureNet


class Flow3DModel(BaseModel):
    """
    FlowNet3D consists of three main blocks:
        1. PointFeatureNet:
            Only uses SetConvLayer (s. below) to obtain a down sampled and more informative
            feature representation of the point cloud. This block uses two SetConvLayer and both point clouds are passed
            through the same PointFeatureNet separately.
        2. PointMixtureNet:
            Uses FlowEmbeddingLayer and SetConvLayer (s. below). One FlowEmbeddingLayer is used to merge both
            point clouds, afterwards two SetConvLayers are used to again down-sample the combined point cloud and
            to obtain a more informative feature representation.
        3. FlowRefinement:
            Only uses SetUpConvLayers (s. below) to up-sample the FlowEmbedding.

    SetConvLayer:
        points from the input point cloud are sampled by using farthest point sampling,
        these sampled points are called regions. For each region all points within a given
        radius r are aggregated by using element-wise max pooling.
    FlowEmbeddingLayer:
        Has both point clouds as input and samples for each point (region) in the first point cloud all
        points from the second point cloud, which are within a given radius r w.r.t. to the region and aggregates
        over both point features by using a element-wise max pooling operation.
    SetUpConvLayer:
        Obtains a source and a target tensor, the source tensor consists of point coordinates and features, where them
        target tensor are only point coordinates. Then for each point (region) in the target tensor, we aggregate over
        the features of all points in the source tensor, which are within a given radius r w.r.t to the region.

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self,
                 learning_rate=1e-6,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999,
                 in_channels=5,
                 n_samples_set_conv=16,
                 n_samples_flow_emb=64,
                 n_samples_set_up_conv=8
                 ):
        super(Flow3DModel, self).__init__()
        self._n_samples_set_conv = n_samples_set_conv
        self._n_samples_flow_emb = n_samples_flow_emb
        self._n_samples_set_up_conv = n_samples_set_up_conv
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams

        self._point_feature_net = PointFeatureNet(in_channels=in_channels, n_samples=self._n_samples_set_conv)
        self._point_mixture = PointMixtureNet(n_samples=self._n_samples_flow_emb)
        self._flow_refinement = FlowRefinementNet(in_channels=512, n_samples=self._n_samples_set_up_conv)
        self._fc = torch.nn.Linear(in_features=128, out_features=3)

    def forward(self, x):
        """
        The usual forward pass function of a torch module
        Both points clouds are passed trough the PointFeatureNet, then both point clouds are combined
        by the PointMixtureNet and at the end the combined point cloud is up-sampled to obtain features
        for each point in the original. The flow is obtained by using a fully-connected layer.
        :param x:
        :return:
        """
        previous_batch, current_batch = x
        previous_batch_pc = previous_batch[0]
        current_batch_pc = current_batch[0]

        
        cuda = torch.device('cuda')
        previous_batch_pc = previous_batch_pc.cuda()
        current_batch_pc = current_batch_pc.cuda()
        previous_batch_pc = previous_batch_pc[:, :, :3]
        current_batch_pc = current_batch_pc[:, :, :3]
        print(previous_batch_pc.shape)

        batch_size, n_points_prev, _ = previous_batch_pc.shape
        batch_size, n_points_cur, _ = current_batch_pc.shape

        # All outputs of the following layers are tuples of (pos, features)
        # --- Point Feature Part ---


        _, _, pf_prev_3 = self._point_feature_net(previous_batch_pc.float())
        pf_curr_1, pf_curr_2, pf_curr_3 = self._point_feature_net(current_batch_pc.float())

        # --- Flow Embedding / Point Mixture Part ---
        _, fe_2, fe_3 = self._point_mixture(x1=pf_prev_3, x2=pf_curr_3)

        # --- Flow Refinement Part ---
        x = self._flow_refinement(pf_curr_1=pf_curr_1, pf_curr_2=pf_curr_2, pf_curr_3=pf_curr_3, fe_2=fe_2, fe_3=fe_3)

        # --- Final fully connected layer ---
        pos, features = x
        features = features.transpose(1, 2)
        x = self._fc(features)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Method to add all command line arguments specific to this module.
        Used to dynamically add the correct arguments required.
        :param parent_parser: The current argparser to add the options to
        :return: the new argparser with the new options
        """
        # Add parameters of all models
        parent_parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parent_parser = BaseModel.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("FlowNet3D")
        parser.add_argument('--n_samples_set_conv', default=16, type=int, help="FlowNet3D specific")
        parser.add_argument('--n_samples_flow_emb', default=64, type=int, help="FlowNet3D specific")
        parser.add_argument('--n_samples_set_up_conv', default=8, type=int, help="FlowNet3D specific")
        return parent_parser
