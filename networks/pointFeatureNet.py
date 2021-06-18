import torch


class PointFeatureNet(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PointFeatureNet, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.batch_norm = torch.nn.BatchNorm1d(out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Encode all points into their embeddings
        :param x: (n_points, in_features)
        :return: (n_points, out_features)
        """
        # linear transformation
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x