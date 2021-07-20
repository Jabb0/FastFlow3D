import math

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def plot_pillars(indices, x_max, x_min, y_max, y_min, grid_cell_size):
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")

    n_pillars_x = math.floor((x_max - x_min) / grid_cell_size)
    n_pillars_y = math.floor((y_max - y_min) / grid_cell_size)
    pillar_matrix = np.zeros(shape=(n_pillars_x, n_pillars_y, 1))

    for x, y in indices:
        pillar_matrix[x, y] += 1

    x_pos, y_pos, z_pos = [], [], []
    x_size, y_size, z_size = [], [], []

    for i in range(pillar_matrix.shape[0]):
        for j in range(pillar_matrix.shape[1]):
            x_pos.append(i * grid_cell_size)
            y_pos.append(j * grid_cell_size)
            z_pos.append(0)

            x_size.append(grid_cell_size)
            y_size.append(grid_cell_size)
            z_size.append(int(pillar_matrix[i, j]))

    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size)
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


def visualize_flows(vis, points, flows):
    """
    Visualize a 3D point cloud where is point is flow-color-coded
    :param vis: visualizer created with open3D, for example:

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    :param points: (n_points, 3)
    :param flows: (n_points, 3)
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(flows)
    #vis.destroy_window()




