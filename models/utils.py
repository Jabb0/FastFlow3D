import torch


def init_weights(m) -> None:
    """
    Apply the weight initialization to a single layer.
    Use this with your_module.apply(init_weights).
    The single layer is a module that has the weights parameter. This does not yield for all modules.
    :param m: the layer to apply the init to
    :return: None
    """
    if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
        # Note: There is also xavier_normal_ but the paper does not state which one they used.
        torch.nn.init.xavier_uniform_(m.weight)


def augment_index(indices, number_features, n_pillars_x):
    indices = indices[:, :, 0] * n_pillars_x + indices[:, :, 1]
    # The indices also need to have a 3D dimension. That dimension is the feature dimension of the inputs.
    # We need to repeat the grid cell index such that all feature dimensions are summed up.
    # Note: we use .expand here. Expand does not actually create new memory.
    #   It just views the same entry multiple times.
    #   But as the index does not need a grad nor is changed in place this is fine.
    # First unsqueeze the (batch, points) vector to (batch, points, 1)
    # Then expand the 1 dimension such that the index is defined for all the features desired
    # -1 means do not change this dim. (batch, points,
    return indices.unsqueeze(-1).expand(-1, -1, number_features)
