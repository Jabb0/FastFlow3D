import torch


class PillarFeatureNet(torch.nn.Module):
    """
    Transform the raw point cloud data of shape (n_points, 3) into a pillar representation of shape (D, P, N).
    D: (x, y, z, x_c, y_c, z_c, x_p, y_p), where x, y and z are the point coords.,  x_c, y_c and z_c are the are
     arithmetic mean of all points in the pillar and x_p, y_p are the offset from the pillar x,y center.
    P: Non-empty pillars per sample (point-cloud).
    N: Number of points per pillar.

    If P or N are too large, pillars/points are sampled randomly.
    If P or N are too small, zero padding is applied.

    References
    ----------
    .. [PointPillars] Alex H. Lang and Sourabh Vora and  Holger Caesar and Lubing Zhou and Jiong Yang and Oscar Beijbom
       PointPillars: Fast Encoders for Object Detection from Point Clouds
       https://arxiv.org/pdf/1812.05784.pdf
    """
    def __init__(self, grid_size=(0.4, 0.4)):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, points):
        """ Input must be a point cloud of shape (n_points, 3) """
        pass

