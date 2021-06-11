from utils.pillars import create_pillars
import numpy as np

def test_create_pillars():
    grid_size = 1

    # create points in range of (0, 0) and (3, 3)
    points = np.array([
        [0, 0, 1],
        [0, 0.1, 2],
        [0, 1, 1],
        [2.9, 2.9, 3],
        [2.9, 2.9, 3],
        [1, 2, 1]
    ])

    pillar_matrix = create_pillars(points, grid_size=grid_size)

    rows = len(pillar_matrix)
    cols = len(pillar_matrix[0])


    for i in range(rows):
        for j in range(cols):
            pillar_matrix[i][j] = len(pillar_matrix[i][j])

    true_pillar_matrix = [
        [2, 1, 0],
        [0, 0, 1],
        [0, 0, 2]
    ]

    assert true_pillar_matrix == pillar_matrix


if __name__ == '__main__':
    test_create_pillars()