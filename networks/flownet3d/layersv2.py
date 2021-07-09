from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.flownet3d.utils.pointnet2_utils import QueryAndGroup, GroupAll, gather_operation, furthest_point_sample
from networks.flownet3d.util import build_shared_mlp


class SetConvLayerV2(nn.Module):
    """
    SetConvLayer
    """
    def __init__(self, sample_rate: float, radius: float, n_samples: int, mlp: List[int],
                 bn: bool = True, use_xyz: bool = True):
        super(SetConvLayerV2, self).__init__()

        self.sample_rate = sample_rate
        self.grouper = QueryAndGroup(radius, n_samples, use_xyz=use_xyz) \
            if sample_rate is not None else GroupAll(use_xyz)

        if use_xyz:
            mlp[0] += 3

        self.mlp = build_shared_mlp(mlp, bn)

    def forward(self, pos: torch.Tensor, features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        pos : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, n_points, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlp[k][-1]), n_points) tensor of the new_features descriptors
        """
        n_points = int(pos.shape[1] * self.sample_rate)

        xyz_flipped = pos.transpose(1, 2).contiguous()
        new_xyz = (
            gather_operation(xyz_flipped, furthest_point_sample(pos, n_points)).transpose(1, 2).contiguous()
            if self.sample_rate is not None else None
        )

        new_features = self.grouper(pos, new_xyz, features)  # (B, C, n_points, n_samples)

        new_features = self.mlp(new_features)  # (B, mlp[-1], n_points, n_samples)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], n_points, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], n_points)

        return new_xyz, new_features


class FlowEmbeddingLayerV2(nn.Module):
    """
    SetConvLayer
    """
    def __init__(self, sample_rate: float, radius: float, n_samples: int, mlp: List[int],
                 bn: bool = True, use_xyz: bool = True):
        super(FlowEmbeddingLayerV2, self).__init__()

        self.sample_rate = sample_rate
        self.grouper = QueryAndGroup(radius, n_samples, use_xyz=use_xyz)\
            if sample_rate is not None else GroupAll(use_xyz)

        if use_xyz:
            mlp[0] += 3

        self.mlp = build_shared_mlp(mlp, bn)

    def forward(self, pos1: torch.Tensor, features1: Optional[torch.Tensor],
                pos2: torch.Tensor, features2: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        pos : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, n_points, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlp[k][-1]), n_points) tensor of the new_features descriptors
        """

        new_features = self.grouper(pos1, pos2, features1, features2)  # (B, C, n_points, n_samples)

        new_features = self.mlp(new_features)  # (B, mlp[-1], n_points, n_samples)
        new_features = F.max_pool2d(
            new_features, kernel_size=[1, new_features.size(3)]
        )  # (B, mlp[-1], n_points, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], n_points)

        return pos2, new_features


class SetConvUpLayerV2(nn.Module):
    """
    SetConvLayer
    """
    def __init__(self, sample_rate: float, radius: float, n_samples: int, mlp: List[int],
                 bn: bool = True, use_xyz: bool = True):
        super(SetConvUpLayerV2, self).__init__()

        self.sample_rate = sample_rate
        self.grouper = QueryAndGroup(radius, n_samples, use_xyz=use_xyz) \
            if sample_rate is not None else GroupAll(use_xyz)

        if use_xyz:
            mlp[0] += 3

        self.mlp = build_shared_mlp(mlp, bn)

    def forward(self, src_pos: torch.Tensor, src_features: Optional[torch.Tensor],
                target_pos: torch.Tensor, target_features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        pos : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, n_points, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlp[k][-1]), n_points) tensor of the new_features descriptors
        """

        new_features = self.grouper(src_pos, target_pos, src_features)  # (B, C, n_points, n_samples)

        new_features = self.mlp(new_features)  # (B, mlp[-1], n_points, n_samples)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], n_points, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], n_points)

        return target_pos, new_features