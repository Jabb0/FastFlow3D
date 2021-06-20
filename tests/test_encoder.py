import torch

def test_encoder():
    """ Check if view is correctly well computed """
    n_pillars = 5 # 5x5
    n_features = 4

    example_array = torch.arange(n_pillars*n_pillars*n_features)
    grid = torch.reshape(example_array, (n_pillars*n_pillars, n_features))
    # grid3d = grid.view((n_features, n_pillars, n_pillars)).long()
    grid3d = grid.view((n_pillars, n_pillars, n_features)).long()
    grid3d = grid3d.permute((2, 0, 1))
    print(grid3d[:, 0, 0])

    # Compute manually the 3D volume from the 2D grid
    manual_3d = torch.zeros(grid3d.shape).long()
    for i in range(grid3d.shape[1]):
        for j in range(grid3d.shape[2]):
            manual_3d[:, i, j] = grid[i * n_pillars + j, :]
    assert torch.equal(manual_3d, grid3d)

    # And now the ungrid operation
    grid2d = grid3d.reshape((n_features, n_pillars * n_pillars))
    grid2d = grid2d.permute((1, 0))
    assert torch.equal(grid, grid2d)


if __name__ == '__main__':
    test_encoder()