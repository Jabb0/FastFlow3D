from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from typing import List
import torch


def make_mlp(in_channels: int, mlp_channels: List[int], batch_norm: bool = True):
    """ Creates a neural network with one linear layer for each entry in mlp_channels
        in_channels is an integer, which defines the number of channels of the input to the neural network.
        mlp_channels is a list of ints, where each int denotes the output features of the corresponding linear layer.
     """
    assert len(mlp_channels) >= 1
    layers = []

    for c in mlp_channels:
        layers += [Linear(in_channels, c)]
        if batch_norm:
            layers += [BatchNorm1d(c)]
        layers += [ReLU()]

        in_channels = c

    return Sequential(*layers)


def transform_data(pc):
    """
    Transforms each point in the given point cloud of shape (batch_size, n_points, 8)
    from (cx, cy, cz,  Δx, Δy, Δz, l0, l1) to (x, y, z, l0, l1)
    """
    x = pc[:, :, 3] + pc[:, :, 0]
    y = pc[:, :, 4] + pc[:, :, 1]
    z = pc[:, :, 5] + pc[:, :, 2]
    l1 = pc[:, :, 6]
    l2 = pc[:, :, 7]
    return torch.stack([x, y, z, l1, l2], dim=2)
