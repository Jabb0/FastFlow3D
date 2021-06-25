from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from typing import List


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
