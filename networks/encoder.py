import torch
import math


class PillarFeatureNet(torch.nn.Module):
    """
    Transform the raw point cloud data of shape (n_points, 3) into a representation of shape (n_points, 6).
    Each point consists of 6 features: (x_c, y_c, z_c, x_delta, y_delta, z_delta). TODO missing 2D of laser data.
    x_c, y_c, z_c: Center of the pillar to which the point belongs.
    x_delta, y_delta, z_delta: Offset from the pillar center to the point.

    References
    ----------
    .. [PointPillars] Alex H. Lang and Sourabh Vora and  Holger Caesar and Lubing Zhou and Jiong Yang and Oscar Beijbom
       PointPillars: Fast Encoders for Object Detection from Point Clouds
       https://arxiv.org/pdf/1812.05784.pdf
    """
    def __init__(self, n_pillars_x, n_pillars_y, in_features=6, out_features=64):
        super().__init__()
        self.n_pillars_x = n_pillars_x
        self.n_pillars_y = n_pillars_y

        self.out_features = out_features

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.batch_norm = torch.nn.BatchNorm1d(out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x, indices):
        """ Input must be the augmented point cloud of shape (n_points, 6) """

        # linear transformation
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # Snap-to-grid
        grid = torch.zeros(size=(self.n_pillars_x, self.n_pillars_y, self.out_features))
        for i in range(x.shape[0]):
            x_idx, y_idx = indices[i].long()
            grid[x_idx, y_idx].add(x[i])

        return grid
