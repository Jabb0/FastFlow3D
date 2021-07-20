from typing import List

import torch


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


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            # TODO is this better than Linear?
            torch.nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(torch.nn.BatchNorm2d(mlp_spec[i]))
        layers.append(torch.nn.ReLU(True))

    return torch.nn.Sequential(*layers)
