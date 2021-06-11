import numpy as np
import math


class Pillar:
    def __init__(self, x, y, grid_size):
        # Lower left point of the pillar rectangle
        self.x = x
        self.y = y

        # Center point of the pillar
        self.x_c = x + grid_size/2
        self.y_c = y + grid_size/2

        self.grid_size = grid_size

        self.points = list()

    def add_point(self, point):
        self.points.append(point)

    def __len__(self):
        return len(self.points)


def create_pillars(pc, grid_size):
    """ If points on border will fall into the next pillar """
    x_max = np.max(pc[:, 0])
    x_min = np.min(pc[:, 0])

    y_max = np.max(pc[:, 1])
    y_min = np.min(pc[:, 1])

    # Get number of pillars in x and y direction
    n_pillars_x = math.floor((x_max - x_min) / grid_size)
    n_pillars_y = math.floor((y_max - y_min) / grid_size)

    # Create 2d matrix, where each entry contains a pillar
    pillar_matrix = [
        [Pillar(x=x_min + j*grid_size, y=y_min + i*grid_size, grid_size=grid_size) for i in range(n_pillars_y + 1)]
        for j in range(n_pillars_x + 1)]

    for point in pc:
        x, y, z = point
        pillar_idx_x = math.floor((x - x_min) / grid_size)
        pillar_idx_y = math.floor((y - y_min) / grid_size)

        pillar_matrix[pillar_idx_x][pillar_idx_y].add_point(point=point)

    return pillar_matrix

