import torch

from models.BaseModel import BaseModel

from networks.flownet3d.pointFeatureNet import PointFeatureNet
from networks.flownet3d.pointMixture import PointMixtureNet
from networks.flownet3d.flowRefinement import FlowRefinementNet
from networks.flownet3d.util import transform_data


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
                 adam_beta_2=0.999):
        super(Flow3DModel, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams

        self._point_feature_net = PointFeatureNet(in_channels=5)
        self._point_mixture = PointMixtureNet()
        self._flow_refinement = FlowRefinementNet(in_channels=512 + 3)
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
        previous_batch_pc, _, previous_batch_mask = previous_batch
        current_batch_pc, _, current_batch_mask = current_batch

        # transform each point from (cx, cy, cz,  Δx, Δy, Δz, l0, l1) to (x, y, z, l0, l1)
        # TODO: Maybe we should change the dataloader to skip pillarization
        #  and use a third collate function that does not process the grid indices
        #   Otherwise these tensor operations might have additional memory
        #   consumption because of the torch computation graph?
        previous_batch_pc = transform_data(previous_batch_pc)
        current_batch_pc = transform_data(current_batch_pc)

        batch_size, n_points_prev, _ = current_batch_pc.shape

        # --- Point Feature Part ---
        _, _, pf_prev_3 = self._point_feature_net(previous_batch_pc.float(), previous_batch_mask)
        pf_curr_1, pf_curr_2, pf_curr_3 = self._point_feature_net(current_batch_pc.float(), current_batch_mask)
        # --- Flow Embedding / Point Mixture Part ---
        _, fe_2, fe_3 = self._point_mixture(x1=pf_prev_3, x2=pf_curr_3)  # NOTE: x2 must be cur point cloud
        # --- Flow Refinement Part ---
        x = self._flow_refinement(pf_curr_1=pf_curr_1, pf_curr_2=pf_curr_2, pf_curr_3=pf_curr_3, fe_2=fe_2, fe_3=fe_3)

        # --- Final fully connected layer ---
        features, pos, batch = x
        x = self._fc(features)

        x = x.view(batch_size, n_points_prev, 3)
        return x


# class Flow3DKaolinModel(BaseModel):
#     """
#     FlowNet3D consists of three main blocks:
#         1. PointFeatureNet:
#             Only uses SetConvLayer (s. below) to obtain a down sampled and more informative
#             feature representation of the point cloud. This block uses two SetConvLayer and both point clouds are passed
#             through the same PointFeatureNet separately.
#         2. PointMixtureNet:
#             Uses FlowEmbeddingLayer and SetConvLayer (s. below). One FlowEmbeddingLayer is used to merge both
#             point clouds, afterwards two SetConvLayers are used to again down-sample the combined point cloud and
#             to obtain a more informative feature representation.
#         3. FlowRefinement:
#             Only uses SetUpConvLayers (s. below) to up-sample the FlowEmbedding.
#
#     SetConvLayer:
#         points from the input point cloud are sampled by using farthest point sampling,
#         these sampled points are called regions. For each region all points within a given
#         radius r are aggregated by using element-wise max pooling.
#     FlowEmbeddingLayer:
#         Has both point clouds as input and samples for each point (region) in the first point cloud all
#         points from the second point cloud, which are within a given radius r w.r.t. to the region and aggregates
#         over both point features by using a element-wise max pooling operation.
#     SetUpConvLayer:
#         Obtains a source and a target tensor, the source tensor consists of point coordinates and features, where them
#         target tensor are only point coordinates. Then for each point (region) in the target tensor, we aggregate over
#         the features of all points in the source tensor, which are within a given radius r w.r.t to the region.
#
#     References
#     ----------
#     .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
#        https://arxiv.org/pdf/1806.01411.pdf
#     """
#     def __init__(self,
#                  learning_rate=1e-6,
#                  adam_beta_1=0.9,
#                  adam_beta_2=0.999):
#         super(Flow3DKaolinModel, self).__init__()
#         self.save_hyperparameters()  # Store the constructor parameters into self.hparams
#
#         from networks.flownet3d.layersv2 import SetConv, FlowEmbedding, SetUpConv
#         self.set_conv1 = SetConv(1024, 0.5, 16, 3, [32, 32, 64])
#         self.set_conv2 = SetConv(256, 1.0, 16, 64, [64, 64, 128])
#         self.flow_embedding = FlowEmbedding(64, 128, [128, 128, 128])
#         self.set_conv3 = SetConv(64, 2.0, 8, 128, [128, 128, 256])
#         self.set_conv4 = SetConv(16, 4.0, 8, 256, [256, 256, 512])
#         self.set_upconv1 = SetUpConv(8, 512, 256, [], [256, 256])
#         self.set_upconv2 = SetUpConv(8, 256, 256, [128, 128, 256], [256])
#         self.set_upconv3 = SetUpConv(8, 256, 64, [128, 128, 256], [256])
#
#     def forward(self, x):
#         """
#         The usual forward pass function of a torch module
#         Both points clouds are passed trough the PointFeatureNet, then both point clouds are combined
#         by the PointMixtureNet and at the end the combined point cloud is up-sampled to obtain features
#         for each point in the original. The flow is obtained by using a fully-connected layer.
#         :param x:
#         :return:
#         """
#         previous_batch, current_batch = x
#         previous_batch_pc, _, previous_batch_mask = previous_batch
#         current_batch_pc, _, current_batch_mask = current_batch
#
#         print(previous_batch_pc.shape)
#         print(previous_batch_mask.shape)
#
#         # transform each point from (cx, cy, cz,  Δx, Δy, Δz, l0, l1) to (x, y, z, l0, l1)
#         previous_batch_pc = transform_data(previous_batch_pc)
#         current_batch_pc = transform_data(current_batch_pc)
#
#         batch_size, n_points_1, _ = previous_batch_pc.shape
#         batch_size, n_points_2, _ = current_batch_pc.shape
#
#         # get features
#         features1 = previous_batch_pc[:, :, 3:]
#         # get pos
#         points1 = previous_batch_pc[:, :, :3]
#
#         # get features
#         features2 = current_batch_pc[:, :, 3:]
#         # get pos
#         points2 = current_batch_pc[:, :, :3]
#
#         points1_1, features1_1 = self.set_conv1(points1, features1)
#         points1_2, features1_2 = self.set_conv2(points1_1, features1_1)
#         points2_1, features2_1 = self.set_conv1(points2, features2)
#         points2_2, features2_2 = self.set_conv2(points2_1, features2_1)
#
#         embedding = self.flow_embedding(points1_2, points2_2, features1_2, features2_2)
#
#         points1_3, features1_3 = self.set_conv3(points1_2, embedding)
#         points1_4, features1_4 = self.set_conv4(points1_3, features1_3)
#
#         new_features1_3 = self.set_upconv1(points1_4, points1_3, features1_4, features1_3)
#         new_features1_2 = self.set_upconv2(points1_3, points1_2, new_features1_3,
#                                            torch.cat([features1_2, embedding], dim=1))
#         new_features1_1 = self.set_upconv3(points1_2, points1_1, new_features1_2, features1_1)
#
#         return new_features1_1
