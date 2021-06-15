import numpy as np
import math


def create_pillars(pc, grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Returns all points with augmented representation and with their corresponding pillar indices.
    Pillar indices consists of a x and y coordinate, which tells to which pillar the point belongs.
    """
    points = list()
    indices = list()

    # Add points to pillars
    for point in pc:
        x, y, z = point

        # Skip points, which are not in the given range.
        if x < x_min or x >= x_max or y < y_min or y >= y_max or z < z_min or z >= z_max:
            continue

        # Compute indices of pillar for current point
        pillar_idx_x = math.floor((x - x_min) / grid_cell_size)
        pillar_idx_y = math.floor((y - y_min) / grid_cell_size)

        # Center of pillar
        pillar_x = x_min + pillar_idx_x * grid_cell_size
        pillar_y = y_min + pillar_idx_y * grid_cell_size
        x_c = pillar_x + grid_cell_size / 2
        y_c = pillar_y + grid_cell_size / 2
        z_c = (z_max - z_min) * 1/2

        # Offset from pillar to current point
        x_delta = x - x_c
        y_delta = y - y_c
        z_delta = z - z_c

        # Add augmented point
        points.append(np.array([x_c, y_c, z_c, x_delta, y_delta, z_delta]))
        # Add indices of the pillar in which the current point lies.
        indices.append([pillar_idx_x, pillar_idx_y])

    return np.array(points), np.array(indices)
