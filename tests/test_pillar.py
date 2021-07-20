import torch
from networks.pillarFeatureNetScatter import PillarFeatureNetScatter


def test_scatter_grid_representation():
    n_points = 5
    batch_size = 1
    n_features = 64
    n_pillars_x = 3
    n_pillars_y = 3
    x = torch.ones(size=(batch_size, n_points, n_features))
    indices = torch.tensor([[0, 0, 0, 0, 0]])  # tensor of shape (batch_size, n_points)
    indices = indices.unsqueeze(-1).expand(-1, -1, n_features)
    pfns = PillarFeatureNetScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y)
    output = pfns(x=x, indices=indices)

    true_output = torch.zeros(size=(batch_size, n_features, n_pillars_x, n_pillars_y))
    # all points are at (0, 0), so all point features are added at this position
    true_output[0, :, 0, 0] = torch.full(size=(n_features, ), fill_value=5)

    assert output.shape == torch.Size((batch_size, n_features, 3, 3))
    assert torch.allclose(output, true_output)

    n_points = 10
    batch_size = 2
    n_features = 64
    n_pillars_x = 5
    n_pillars_y = 5
    x = torch.ones(size=(batch_size, n_points, n_features))
    indices = torch.tensor([[1, 2, 3, 3, 1]])  # tensor of shape (batch_size, n_points)
    indices = indices.unsqueeze(-1).expand(-1, -1, n_features)
    pfns = PillarFeatureNetScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y)
    output = pfns(x=x, indices=indices)

    true_output = torch.zeros(size=(batch_size, n_features, n_pillars_x, n_pillars_y))
    # all points are at (0, 0), so all point features are added at this position
    true_output[0, :, 0, 1] = torch.full(size=(n_features, ), fill_value=2)
    true_output[0, :, 0, 2] = torch.full(size=(n_features, ), fill_value=1)
    true_output[0, :, 0, 3] = torch.full(size=(n_features, ), fill_value=2)

    assert output.shape == torch.Size((batch_size, n_features, 5, 5))
    assert torch.allclose(output, true_output)

    n_points = 10
    batch_size = 2
    n_features = 64
    n_pillars_x = 5
    n_pillars_y = 5
    x = torch.ones(size=(batch_size, n_points, n_features))
    indices = torch.tensor([[1, 2, 3, 3, 1],
                            [1, 2, 6, 3, 1]])  # tensor of shape (batch_size, n_points)
    indices = indices.unsqueeze(-1).expand(-1, -1, n_features)
    pfns = PillarFeatureNetScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y)
    output = pfns(x=x, indices=indices)

    true_output = torch.zeros(size=(batch_size, n_features, n_pillars_x, n_pillars_y))
    # all points are at (0, 0), so all point features are added at this position
    true_output[0, :, 0, 1] = torch.full(size=(n_features, ), fill_value=2)
    true_output[0, :, 0, 2] = torch.full(size=(n_features, ), fill_value=1)
    true_output[0, :, 0, 3] = torch.full(size=(n_features, ), fill_value=2)

    true_output[1, :, 0, 1] = torch.full(size=(n_features, ), fill_value=2)
    true_output[1, :, 0, 2] = torch.full(size=(n_features, ), fill_value=1)
    true_output[1, :, 0, 3] = torch.full(size=(n_features, ), fill_value=1)
    true_output[1, :, 1, 1] = torch.full(size=(n_features, ), fill_value=1)

    assert output.shape == torch.Size((batch_size, n_features, 5, 5))
    assert torch.allclose(output, true_output)


if __name__ == '__main__':
    test_scatter_grid_representation()
