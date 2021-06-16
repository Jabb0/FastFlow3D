import numpy as np
import math


def remove_out_of_bounds_points(pc, y, x_min, x_max, y_min, y_max, z_min, z_max):
    # Calculate the cell id that this entry falls into
    # Store the X, Y indices of the grid cells for each point cloud point
    mask = (pc[:, 0] >= x_min) & (pc[:, 0] <= x_max) \
           & (pc[:, 1] >= y_min) & (pc[:, 1] <= y_max) \
           & (pc[:, 2] >= z_min) & (pc[:, 2] <= z_max)
    pc_valid = pc[mask]
    y_valid = None
    if y is not None:
        y_valid = y[mask]
    return pc_valid, y_valid


def create_pillars_matrix(pc_valid, grid_cell_size, x_min, y_min, z_min, z_max):
    """
    Compute the pillars using matrix operations.
    :param pc: point cloud data. (N_points, features) with the first 3 features being the x,y,z coordinates.
    :return: augmented_pointcloud, grid_cell_indices, y_valid
    """
    num_laser_features = pc_valid.shape[1] - 3  # Calculate the number of laser features that are not the coordinates.

    grid_cell_indices = np.zeros((pc_valid.shape[0], 2), dtype=int)
    grid_cell_indices[:, 0] = ((pc_valid[:, 0] - x_min) / grid_cell_size).astype(int)
    grid_cell_indices[:, 1] = ((pc_valid[:, 1] - y_min) / grid_cell_size).astype(int)

    # Initialize the new pointcloud with 8 features for each point
    augmented_pc = np.zeros((pc_valid.shape[0], 6 + num_laser_features))
    # Set every cell z-center to the same z-center
    augmented_pc[:, 2] = (z_max - z_min) * 1 / 2
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

    # Take the two laser features
    augmented_pc[:, 6:] = pc_valid[:, 3:]

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
        z_c = (z_max - z_min) * 1 / 2

        # Offset from pillar to current point
        x_delta = x - x_c
        y_delta = y - y_c
        z_delta = z - z_c

        # Add augmented point
        points.append(np.array([x_c, y_c, z_c, x_delta, y_delta, z_delta]))
        # Add indices of the pillar in which the current point lies.
        indices.append([pillar_idx_x, pillar_idx_y])

    return np.array(points), np.array(indices)
