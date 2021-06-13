import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from itertools import chain


def plot_pillars_center(pillars, grid_max, grid_min):
    """ pillars must be in shape (n_pillars, 2) """
    x, y = [], []
    for pillar in pillars:
        x.append(pillar.x_c)
        y.append(pillar.y_c)

    plt.plot(x, y, marker='.', color='k', linestyle='none')
    plt.ylim([grid_min, grid_max])
    plt.xlim([grid_min, grid_max])
    plt.show()


def plot_pillars(pc, pillar_matrix, grid_size):
    pillars = list(chain.from_iterable(pillar_matrix))
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")

    x_pos, y_pos, z_pos = [], [], []
    x_size, y_size, z_size = [], [], []
    for p in pillars:
        x_pos.append(p.x)
        y_pos.append(p.y)
        z_pos.append(0)

        x_size.append(grid_size)
        y_size.append(grid_size)
        z_size.append(len(p))

    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, edgecolor='black')

    x, y, z = [], [], []
    for p in pc:
        x.append(p[0])
        y.append(p[1])
        z.append(np.max(z_size) * 1.1)

    ax.scatter3D(x, y, z, color="green")

    plt.show()


def plot_2d_point_cloud(pc):
    fig, ax = plt.subplots(figsize=(15, 15))

    x, y = [], []
    for p in pc:
        x.append(p[0])
        y.append(p[1])
    ax.scatter(x, y, color="green")
    plt.show()


def visualize_point_cloud(points):
    """ Input must be a point cloud of shape (n_points, 3) """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])
