import numpy as np
import math


def create_pillars_matrix(pc, y, grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Experimental code to compute the pillars using matrix operations.
    :param pc: point cloud data
    """
    # Calculate the cell id that this entry falls into
    # Store the X, Y indices of the grid cells for each point cloud point
    mask = (pc[:, 0] >= x_min) & (pc[:, 0] <= x_max) \
            & (pc[:, 1] >= y_min) & (pc[:, 1] <= y_max) \
            & (pc[:, 2] >= z_min) & (pc[:, 2] <= z_max)
    pc_valid = pc[mask]
    y_valid = y[mask]
    grid_cell_indices = np.zeros((pc_valid.shape[0], 2), dtype=int)
    grid_cell_indices[:, 0] = ((pc_valid[:, 0] - x_min) / grid_cell_size).astype(int)
    grid_cell_indices[:, 1] = ((pc_valid[:, 1] - y_min) / grid_cell_size).astype(int)

    # Initialize the new pointcloud with 8 features for each point
    augmented_pc = np.zeros((pc_valid.shape[0], 8))
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
    # l0
    augmented_pc[:, 6] = pc_valid[:, 3]
    # l1
    augmented_pc[:, 7] = pc_valid[:, 4]

    return augmented_pc, grid_cell_indices, y_valid
