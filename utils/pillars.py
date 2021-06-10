import math
import numpy as np


class Pillar:
    def __init__(self, x, y, x_c, y_c, grid_size):
        # Lower left point of the pillar rectangle
        self.x = x
        self.y = y

        # Center point of the pillar
        self.x_c = x_c
        self.y_c = y_c

        self.grid_size = grid_size

        self.points = list()

    def add_point(self, point):
        self.points.append(point)

    def is_empty(self):
        if len(self.points) == 0:
            return True
        else:
            return False

    def point_in_pillar(self, point):
        """ Checks if the given point lies in the rectangle spanned by the pillar. """
        minX = self.x
        maxX = self.x + self.grid_size

        minY = self.y
        maxY = self.y + self.grid_size

        x, y, _ = point
        if minX <= x < maxX:
            if minY <= y < maxY:
                return True
        return False

    def __len__(self):
        return len(self.points)


def create_pillars(cp, grid_size=1):
    x_max = np.max(cp[:, 0])
    x_min = np.min(cp[:, 0])

    y_max = np.max(cp[:, 1])
    y_min = np.min(cp[:, 1])

    # get max value in x and y direction and round to higher int
    grid_max = int(math.ceil(np.max([x_max, y_max])))
    grid_min = np.min([x_min, y_min])  # get max value in x and y
    # round to lower int, if it is negative, otherwise take higher int
    grid_min = int(math.ceil(grid_min)) if grid_min >= 0 else -int(math.ceil(np.abs(grid_min)))

    # Get all centered pillar points
    x_grid = np.arange(grid_min, grid_max, grid_size)
    y_grid = np.arange(grid_min, grid_max, grid_size)

    pillars = list()
    for px in x_grid:
        for py in y_grid:
            pillar = Pillar(x=px, y=py, x_c=px + grid_size / 2, y_c=py + grid_size / 2, grid_size=grid_size)
            pillars.append(pillar)

    # Plot pillar centered points
    # TODO
    # plot_pillars_center(pillars, grid_max, grid_min)

    return pillars


def assign_points_to_pillars(cp, pillars):
    """
    Goes through each point in the point cloud (cp) and assigns each point to the corresponding pillar.
    A point is assigned to a pillar, if the point lies in the rectangle spanned by the pillar.
    """
    for point in cp:
        for p in pillars:
            if p.point_in_pillar(point):
                p.add_point(point)
    return pillars
