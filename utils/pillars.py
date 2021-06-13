import numpy as np
import math


class Pillar:
    def __init__(self, x, y, grid_cell_size, z_min, z_max):
        # Lower left point of the pillar rectangle
        self.x = x
        self.y = y

        # Center point of the pillar
        self.x_c = x + grid_cell_size/2
        self.y_c = y + grid_cell_size/2
        self.z_c = (z_max - z_min) * 1/2

        # Size of the grid cell/pillar (x and y direction have the same size)
        self.grid_cell_size = grid_cell_size

        # List of points of this pillar.
        self.points = list()

    def add_point(self, point):
        x, y, z = point
        x_delta = x - self.x_c
        y_delta = y - self.y_c
        z_delta = z - self.z_c
        self.points.append(np.array([self.x_c, self.y_c, self.z_c, x_delta, y_delta, z_delta]))

    def augment_point_representation(self):
        """"
        Augments each point by
        (x_c, y_c, z_c): Center of the pillar.
        (x_delta, y_delta, z_delta): Offset from pillar center to the point
        """
        for i in range(len(self.points)):
            x, y, z = self.points[i]
            x_delta = x - self.x_c
            y_delta = y - self.y_c
            z_delta = z - self.z_c
            self.points[i] = np.array([self.x_c, self.y_c, self.z_c, x_delta, y_delta, z_delta])

    def __len__(self):
        return len(self.points)


def create_pillars(pc, grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max):
    # Get number of pillars in x and y direction
    n_pillars_x = math.floor((x_max - x_min) / grid_cell_size) + 1
    n_pillars_y = math.floor((y_max - y_min) / grid_cell_size) + 1

    points = list()
    indices = list()

    # Init 2D matrix, where each entry contains a pillar
    pillar_matrix = np.empty(shape=(n_pillars_x, n_pillars_y), dtype=Pillar)

    # Add points to pillars
    for point in pc:
        x, y, z = point

        # Skip points, which are not in the given range.
        if x < x_min or x > x_max or y < y_min or y > y_max or z < z_min or z > z_max:
            continue

        # Compute indices of pillar for current point
        pillar_idx_x = math.floor((x - x_min) / grid_cell_size)
        pillar_idx_y = math.floor((y - y_min) / grid_cell_size)

        # Add Pillar if entry is none.
        if pillar_matrix[pillar_idx_x, pillar_idx_y] is None:
            pillar = Pillar(
                x=x_min + pillar_idx_x*grid_cell_size,
                y=y_min + pillar_idx_y*grid_cell_size,
                grid_cell_size=grid_cell_size,
                z_min=z_min,
                z_max=z_max
            )
            pillar_matrix[pillar_idx_x, pillar_idx_y] = pillar
        # Add point to corresponding pillar
        pillar_matrix[pillar_idx_x, pillar_idx_y].add_point(point=point)

        points.append(pillar_matrix[pillar_idx_x, pillar_idx_y].points[-1])
        indices.append([pillar_idx_x, pillar_idx_y])

    return np.array(points), np.array(indices)
