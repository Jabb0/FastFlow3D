from utils.pillars import create_pillars
import numpy as np


def test_create_pillars():
    grid_cell_size = 1

    # create points in range of (0, 0) and (3, 3)
    points = np.array([
        [0, 0, 1],
        [0, 0.1, 2],
        [0, 1, 1],
        [2.9, 2.9, 3],
        [2.9, 2.9, 3],
        [1, 2, 1]
    ])

    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])

    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])

    z_max = np.max(points[:, 2])
    z_min = np.min(points[:, 2])

    points, indices = create_pillars(points, grid_cell_size=grid_cell_size, x_min=x_min, x_max=x_max,
                                     y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)

    true_indices = np.array([[0, 0],
                             [0, 0],
                             [0, 1],
                             [2, 2],
                             [2, 2],
                             [1, 2]]
                            )

    assert np.all(indices == true_indices)


if __name__ == '__main__':
    test_create_pillars()
