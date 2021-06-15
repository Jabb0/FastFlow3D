import numpy as np
import math


def create_pillars_matrix(pc, grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Experimental code to compute the pillars using matrix operations.
    :param pc: point cloud data
    """
    # Calculate the cell id that this entry falls into
    # Store the X, Y indices of the grid cells for each point cloud point
    pc_valid = pc[(pc[:, 0] >= x_min) & (pc[:, 0] <= x_max)
                  & (pc[:, 1] >= y_min) & (pc[:, 1] <= y_max)
                  & (pc[:, 2] >= z_min) & (pc[:, 2] <= z_max)]
    grid_cell_indices = np.zeros((pc_valid.shape[0], 2), dtype=int)
    grid_cell_indices[:, 0] = ((pc_valid[:, 0] - x_min) / grid_cell_size).astype(int)
    grid_cell_indices[:, 1] = ((pc_valid[:, 1] - y_min) / grid_cell_size).astype(int)

    # Initialize the new pointcloud with 8 features for each point
    augmented_pc = np.zeros((pc_valid.shape[0], 6))  # TODO has to be 8
    # Set every cell z-center to the same z-center
    augmented_pc[:, 2] = (z_max - z_min) * 1/2
    # Set the x cell center depending on the x cell id of each point
    augmented_pc[:, 0] = x_min + 1 / 2 * grid_cell_size + grid_cell_size * grid_cell_indices[:, 0]
    # Set the y cell center depending on the y cell id of each point
    augmented_pc[:, 1] = y_min + 1 / 2 * grid_cell_size + grid_cell_size * grid_cell_indices[:, 1]

    # Calculate the distance of the point to the center.
    # x
    augmented_pc[:, 3] = pc_valid[:, 0] - augmented_pc[:, 0]
    # y
    augmented_pc[:, 4] = pc_valid[:, 1] - augmented_pc[:, 1]
    # z
    augmented_pc[:, 5] = pc_valid[:, 2] - augmented_pc[:, 2]

    # TODO: Copy the last two features for each point (laser_features 1 and 2) but they are not here right now.
    #  Thus keep them 0.

    return augmented_pc, grid_cell_indices


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
