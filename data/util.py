import torch

from utils.pillars import create_pillars_matrix, remove_out_of_bounds_points
from torch.utils.data._utils.collate import default_collate
import numpy as np


class ApplyPillarization:
    def __init__(self, grid_cell_size, x_min, y_min, z_min, z_max, n_pillars_x):
        self._grid_cell_size = grid_cell_size
        self._z_max = z_max
        self._z_min = z_min
        self._y_min = y_min
        self._x_min = x_min
        self._n_pillars_x = n_pillars_x

    """ Transforms an point cloud to the augmented pointcloud depending on Pillarization """

    def __call__(self, x):
        point_cloud, grid_indices = create_pillars_matrix(x,
                                                          grid_cell_size=self._grid_cell_size,
                                                          x_min=self._x_min,
                                                          y_min=self._y_min,
                                                          z_min=self._z_min, z_max=self._z_max,
                                                          n_pillars_x=self._n_pillars_x)
        return point_cloud, grid_indices


def drop_points_function(x_min, x_max, y_min, y_max, z_min, z_max):
    def inner(x, y):
        return remove_out_of_bounds_points(x, y,
                                           x_min=x_min,
                                           y_min=y_min,
                                           z_min=z_min,
                                           z_max=z_max,
                                           x_max=x_max,
                                           y_max=y_max
                                           )

    return inner


def custom_collate(batch):
    """
    We need this custom collate because of the structure of our data.
    :param batch:
    :return:
    """
    # Only convert the points clouds from numpy arrays to tensors
    batch_previous = [
        [torch.as_tensor(e) for e in entry[0][0]] for entry in batch
    ]
    batch_current = [
        [torch.as_tensor(e) for e in entry[0][1]] for entry in batch
    ]

    # For the targets we can only transform each entry to a tensor and not stack them
    batch_targets = [
        torch.as_tensor(entry[1]) for entry in batch
    ]

    return (batch_previous, batch_current), batch_targets


# ------------- Preprocessing Functions ---------------


def get_coordinates_and_features(point_cloud, transform=None):
    """
    Parse a point clound into coordinates and features.
    :param point_cloud: Full [N, 9] point cloud
    :param transform: Optional parameter. Transformation matrix to apply
    to the coordinates of the point cloud
    :return: [N, 5] where N is the number of points and 5 is [x, y, z, intensity, elongation]
    """
    points_coord, features, flows = point_cloud[:, 0:3], point_cloud[:, 3:5], point_cloud[:, 5:]
    if transform is not None:
        ones = np.ones((points_coord.shape[0], 1))
        points_coord = np.hstack((points_coord, ones))
        points_coord = transform @ points_coord.T
        points_coord = points_coord[0:-1, :]
        points_coord = points_coord.T
    point_cloud = np.hstack((points_coord, features))
    return point_cloud


def _pad_batch(batch):
    # Get the number of points in the largest point cloud
    true_number_of_points = [e[1].shape[0] for e in batch]
    max_points_prev = np.max(true_number_of_points)

    # We need a mask of all the points that actually exist
    zeros = np.zeros((len(batch), max_points_prev), dtype=bool)
    # Mark all points that ARE NOT padded
    for i, n in enumerate(true_number_of_points):
        zeros[i, :n] = 1

    # resize all tensors to the max points size
    # Use np.pad to perform this action. Do not pad the second dimension and pad the first dimension AFTER only
    return [
        [
            np.pad(entry[0], ((0, max_points_prev - entry[0].shape[0]), (0, 0))),
            np.pad(entry[1], (0, max_points_prev - entry[1].shape[0])),
            zeros[i]
        ] for i, entry in enumerate(batch)
    ]


def _pad_targets(batch):
    true_number_of_points = [e.shape[0] for e in batch]
    max_points = np.max(true_number_of_points)
    return [
        np.pad(entry, ((0, max_points - entry.shape[0]), (0, 0)))
        for entry in batch
    ]


def custom_collate_batch(batch):
    """
    This version of the collate function create the batch necessary for the input to the network.

    Take the list of entries and batch them together.
        This means a batch of the previous images and a batch of the current images and a batch of flows.
    Because point clouds have different number of points the batching needs the points clouds with less points
        being zero padded.
    Note that this requires to filter out the zero padded points later on.

    :param batch: batch_size long list of ((prev, cur), flows) pointcloud tuples with flows.
        prev and cur are tuples of (point_cloud, grid_indices, mask)
         point clouds are (N_points, features) with different N_points each
    :return: ((batch_prev, batch_cur), batch_flows)
    """
    # Build numpy array with data

    # Only convert the points clouds from numpy arrays to tensors
    # entry[0, 0] is the previous (point_cloud, grid_index) entry
    batch_previous = [
        entry[0][0] for entry in batch
    ]
    batch_previous = _pad_batch(batch_previous)

    batch_current = [
        entry[0][1] for entry in batch
    ]
    batch_current = _pad_batch(batch_current)

    # For the targets we can only transform each entry to a tensor and not stack them
    batch_targets = [
        entry[1] for entry in batch
    ]
    batch_targets = _pad_targets(batch_targets)

    # Call the default collate to stack everything
    batch_previous = default_collate(batch_previous)
    batch_current = default_collate(batch_current)
    batch_targets = default_collate(batch_targets)

    # Return a tensor that consists of
    # the data batches consist of batches of tensors
    #   1. (batch_size, max_n_points, features) the point cloud batch
    #   2. (batch_size, max_n_points) the 1D grid_indices encoding to map to
    #   3. (batch_size, max_n_points) the 0-1 encoding if the element is padded
    # Batch previous for the previous frame
    # Batch current for the current frame

    # The targets consist of
    #   (batch_size, max_n_points, target_features). should by 4D x,y,z flow and class id

    return (batch_previous, batch_current), batch_targets
