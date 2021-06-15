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
