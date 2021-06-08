"""
Simple dense two layer
"""
import torch


class TwoLayerNet(torch.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(TwoLayerNet, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        The usual forward pass function of a torch module
        :param x:
        :return:
        """
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x