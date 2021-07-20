import torch

import re
import os
import glob

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
    true_number_of_points = [e[0].shape[0] for e in batch]
    max_points_prev = np.max(true_number_of_points)

    # We need a mask of all the points that actually exist
    zeros = np.zeros((len(batch), max_points_prev), dtype=bool)
    # Mark all points that ARE NOT padded
    for i, n in enumerate(true_number_of_points):
        zeros[i, :n] = 1

    # resize all tensors to the max points size
    # Use np.pad to perform this action. Do not pad the second dimension and pad the first dimension AFTER only
    return [
        [np.pad(entry[0], ((0, max_points_prev - entry[0].shape[0]), (0, 0))),
         np.pad(entry[1], (0, max_points_prev - entry[1].shape[0])) if entry[1] is not None
         else np.empty(shape=(max_points_prev, )),  # set empty array, if there is None entry in the tuple
         # (for baseline, we do not have grid indices, therefore this tuple entry is None)
         zeros[i]] for i, entry in enumerate(batch)
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


def load_pfm(filename):
    file = open(filename, 'r', newline='', encoding='latin-1')

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    file.close()

    return np.reshape(data, shape), scale


def bilinear_interp_val(vmap, y, x):
    """
    bilinear interpolation on a 2D map
    """
    h, w = vmap.shape
    x1 = int(x)
    x2 = x1 + 1
    x2 = w - 1 if x2 > (w - 1) else x2
    y1 = int(y)
    y2 = y1 + 1
    y2 = h - 1 if y2 > (h - 1) else y2
    Q11 = vmap[y1, x1]
    Q21 = vmap[y1, x2]
    Q12 = vmap[y2, x1]
    Q22 = vmap[y2, x2]
    return Q11 * (x2 - x) * (y2 - y) + Q21 * (x - x1) * (y2 - y) + Q12 * (x2 - x) * (y - y1) + Q22 * (x - x1) * (y - y1)


def get_3d_pos_xy(y_prime, x_prime, depth, focal_length=1050., w=960, h=540):
    """ depth pop up """
    y = (y_prime - h / 2.) * depth / focal_length
    x = (x_prime - w / 2.) * depth / focal_length
    return [x, y, depth]


def generate_flying_things_point_cloud(fname_disparity, fname_disparity_next_frame, fname_disparity_change,
                                       fname_optical_flow, max_cut=35, focal_length=1050., add_label=True):
    # generate needed data
    disparity_np, _ = load_pfm(fname_disparity)
    disparity_next_frame_np, _ = load_pfm(fname_disparity_next_frame)
    disparity_change_np, _ = load_pfm(fname_disparity_change)
    optical_flow_np, _ = load_pfm(fname_optical_flow)

    depth_np = focal_length / disparity_np
    depth_next_frame_np = focal_length / disparity_next_frame_np
    future_depth_np = focal_length / (disparity_np + disparity_change_np)
    h, w = disparity_np.shape

    try:
        depth_requirement = depth_np < max_cut
    except:
        return None

    satisfy_pix1 = np.column_stack(np.where(depth_requirement))
    sampled_pix1_x = satisfy_pix1[:, 1]
    sampled_pix1_y = satisfy_pix1[:, 0]
    n_1 = sampled_pix1_x.shape[0]
    current_pos1 = np.array([get_3d_pos_xy(sampled_pix1_y[i], sampled_pix1_x[i],
                                           depth_np[int(sampled_pix1_y[i]),
                                                    int(sampled_pix1_x[i])]) for i in range(n_1)])

    # sampled_optical_flow_x = np.array(
    #     [optical_flow_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])][0] for i in range(n_1)])
    # sampled_optical_flow_y = np.array(
    #     [optical_flow_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])][1] for i in range(n_1)])
    # future_pix1_x = sampled_pix1_x + sampled_optical_flow_x
    # future_pix1_y = sampled_pix1_y - sampled_optical_flow_y
    # future_pos1 = np.array([get_3d_pos_xy(future_pix1_y[i], future_pix1_x[i],
    #                                       future_depth_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])]) for i in
    #                         range(n_1)])
    # flow = future_pos1 - current_pos1

    try:
        depth_requirement = depth_next_frame_np < max_cut
    except:
        return None

    satisfy_pix2 = np.column_stack(np.where(depth_next_frame_np < max_cut))
    sampled_pix2_x = satisfy_pix2[:, 1]
    sampled_pix2_y = satisfy_pix2[:, 0]
    n_2 = sampled_pix2_x.shape[0]

    current_pos2 = np.array([get_3d_pos_xy(sampled_pix2_y[i], sampled_pix2_x[i],
                                           depth_next_frame_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])]) for i in
                             range(n_2)])

    # TODO Check if this is correct (i would say it is correct, because we use backward optical flow instead of forward)
    sampled_optical_flow_x = np.array(
        [optical_flow_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])][0] for i in range(n_2)])
    sampled_optical_flow_y = np.array(
        [optical_flow_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])][1] for i in range(n_2)])
    future_pix2_x = sampled_pix2_x + sampled_optical_flow_x
    future_pix2_y = sampled_pix2_y - sampled_optical_flow_y
    future_pos2 = np.array([get_3d_pos_xy(future_pix2_y[i], future_pix2_x[i],
                                          future_depth_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])]) for i in
                            range(n_2)])
    flow = future_pos2 - current_pos2

    # mask, judge whether point move out of fov or occluded by other object after motion
    future_pos2_depth = future_depth_np[sampled_pix2_y, sampled_pix2_x]
    future_pos2_foreground_depth = np.zeros_like(future_pos2_depth)
    valid_mask_fov2 = np.ones_like(future_pos2_depth, dtype=bool)
    for i in range(future_pos2_depth.shape[0]):
        if 0 < future_pix2_y[i] < h and 0 < future_pix2_x[i] < w:
            future_pos2_foreground_depth[i] = bilinear_interp_val(depth_next_frame_np, future_pix2_y[i], future_pix2_x[i])
        else:
            valid_mask_fov2[i] = False
    valid_mask_occ2 = (future_pos2_foreground_depth - future_pos2_depth) > -5e-1

    mask2 = valid_mask_occ2 & valid_mask_fov2

    # Add redundant zero labels for each flow in order to fit the shape
    if add_label:
        labelled_flow = np.ones(shape=(flow.shape[0], flow.shape[1] + 1))
        labelled_flow[:, :-1] = flow
        flow = labelled_flow

    return current_pos1, current_pos2, flow, mask2


def get_all_flying_things_frames(input_dir, disp_dir, opt_dir, disp_change_dir):
    all_files_disparity = glob.glob(os.path.join(input_dir, '{0}/*.pfm'.format(disp_dir)))
    all_files_disparity_change = glob.glob(os.path.join(input_dir, '{0}/*.pfm'.format(disp_change_dir)))
    all_files_opt_flow = glob.glob(os.path.join(input_dir, '{0}/*.pfm'.format(opt_dir)))

    assert len(all_files_disparity) == len(all_files_opt_flow) == len(all_files_disparity_change)

    return all_files_disparity, all_files_disparity_change, all_files_opt_flow