import time

import tensorflow as tf
import torch
import numpy as np

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2

from utils.pillars import create_pillars_matrix, remove_out_of_bounds_points
from torch.utils.data._utils.collate import default_collate
import numpy as np
import os, glob
import pickle

from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       point_flows,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
    """Convert range images to point cloud.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
      range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
      (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    flows = []

    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.compat.v1.where(range_image_mask))

        flow = point_flows[c.name][ri_index]
        flow_tensor = tf.reshape(tf.convert_to_tensor(value=flow.data), flow.shape.dims)
        flow_points_tensor = tf.gather_nd(flow_tensor,
                                          tf.compat.v1.where(range_image_mask))

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor,
                                        tf.compat.v1.where(range_image_mask))

        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        flows.append(flow_points_tensor.numpy())

    return points, cp_points, flows


def parse_range_image_and_camera_projection(frame):
    """
    Parse range images and camera projections given a frame.

  Args:
     frame: open dataset frame proto

  Returns:
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
  """
    range_images = {}
    camera_projections = {}
    point_flows = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if len(laser.ri_return1.range_image_flow_compressed) > 0:
                range_image_flow_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_flow_compressed, 'ZLIB')
                ri = dataset_pb2.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_flow_str_tensor.numpy()))
                point_flows[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy()))

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
        if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            if len(laser.ri_return2.range_image_flow_compressed) > 0:
                range_image_flow_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.range_image_flow_compressed, 'ZLIB')
                ri = dataset_pb2.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_flow_str_tensor.numpy()))
                point_flows[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, point_flows, range_image_top_pose


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

def save_point_cloud(compressed_frame, file_path):
    """
    Compute the point cloud from a frame and stores it into disk.
    :param compressed_frame: compressed frame from a TFRecord
    :param file_path: name path that will have the stored point cloud
    :returns:
        - points - [N, 5] matrix which stores the [x, y, z, intensity, elongation] in the frame reference
        - flows - [N, 4] matrix where each row is the flow for each point in the form [vx, vy, vz, label]
                  in the reference frame
        - transform - [,16] flattened transformation matrix
    """
    frame = get_uncompressed_frame(compressed_frame)
    points, flows = compute_features(frame)
    point_cloud = np.hstack((points, flows))
    np.save(file_path, point_cloud)
    transform = list(frame.pose.transform)
    return points, flows, transform


def preprocess(tfrecord_file, output_path, frames_per_segment = None):
    """
    TFRecord file to store in a suitable form for training
    in disk. A point cloud in disk has dimensions [N, 9] where N is the number of points
    and per each point it stores [x, y, z, intensity, elongation, vx, vy, vz, label].
    It stores in a dictionary relevant metadata:
        - look-up table: It has the form [[t_1, t_0], [t_2, t_1], ... , [t_n, t_(n-1)]], where t_i is
        (file_path, transform), where file_path is the file where the point cloud is stored and transform the transformation
        to apply to a point to change it reference frame from global to the car frame in that moment.
        - flows information: min and max flows encountered for then visualize pointclouds properly

    :param tfrecord_file: TFRecord file. It should have the flow extension.
                          They can be downloaded from https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow
    :param output_path: path where the processed point clouds will be saved.
    """
    tfrecord_filename = os.path.basename(tfrecord_file)
    tfrecord_filename = os.path.splitext(tfrecord_filename)[0]

    look_up_table = []
    metadata_path = os.path.join(output_path, f"metadata_{tfrecord_filename}")  # look-up table and flows mins and maxs
    loaded_file = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    previous_frame = None

    # Needed of max and min flows for normalizing in visualization
    min_vx_global, max_vx_global = np.inf, -np.inf
    min_vy_global, max_vy_global = np.inf, -np.inf
    min_vz_global, max_vz_global = np.inf, -np.inf

    for j, frame in enumerate(loaded_file):
        output_file_name = f"pointCloud_file_{tfrecord_filename}_frame_{j}.npy"
        point_cloud_path = os.path.join(output_path, output_file_name)
        # Process frame and store point clouds into disk
        _, flows, pose_transform = save_point_cloud(frame, point_cloud_path)
        # TODO filter invalid flow (label -1)
        min_vx, min_vy, min_vz = flows[:, :-1].min(axis=0)
        max_vx, max_vy, max_vz = flows[:, :-1].max(axis=0)
        min_vx_global = min(min_vx_global, min_vx)
        min_vy_global = min(min_vy_global, min_vy)
        min_vz_global = min(min_vz_global, min_vz)
        max_vx_global = max(max_vx_global, max_vx)
        max_vy_global = max(max_vy_global, max_vy)
        max_vz_global = max(max_vz_global, max_vz)

        if j == 0:
            previous_frame = (output_file_name, pose_transform)
        else:
            current_frame = (output_file_name, pose_transform)
            look_up_table.append([current_frame, previous_frame])
            previous_frame = current_frame
        if j > 5:
            break
        if frames_per_segment is not None and j == frames_per_segment:
            break

    # Save metadata into disk
    flows_info = {'min_vx': min_vx_global,
                  'max_vx': max_vx_global,
                  'min_vy': min_vy_global,
                  'max_vy': max_vy_global,
                  'min_vz': min_vz_global,
                  'max_vz': max_vz_global}

    metadata = {'look_up_table': look_up_table,
                'flows_information': flows_info}

    with open(metadata_path, 'wb') as metadata_file:
        pickle.dump(metadata, metadata_file)

def get_uncompressed_frame(compressed_frame):
    """
    :param compressed_frame: Compressed frame
    :return: Uncompressed frame
    """
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(compressed_frame.numpy()))
    return frame


def compute_features(frame):
    """
    :param frame: Uncompressed frame
    :return: [N, F], [N, 4], where N is the number of points, F the number of features,
    which is [x, y, z, intensity, elongation] and 4 in the second results stands for [vx, vy, vz, label],
    which corresponds to the flow information
    """
    range_images, camera_projections, point_flows, range_image_top_pose = parse_range_image_and_camera_projection(
        frame)

    points, cp_points, flows = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        point_flows,
        range_image_top_pose,
        keep_polar_features=True)

    # 3D points in the vehicle reference frame
    points_all = np.concatenate(points, axis=0)
    flows_all = np.concatenate(flows, axis=0)
    # We skip the range feature since pillars will account for it
    # Note that first are features and then point coordinates
    points_features, points_coord = points_all[:, 1:3], points_all[:, 3:points_all.shape[1]]
    points_all = np.hstack((points_coord, points_features))
    return points_all, flows_all


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


def merge_metadata(input_path):
    """
    Merge individual look-up table and flows mins and maxs and store it in the input_path with the name metadata
    :param input_path: Path with the local metadata in the form metadata_[tfRecordName]
    """
    look_up_table = []
    flows_info = None

    os.chdir(input_path)
    for file in glob.glob("metadata_*"):
        file_name = os.path.abspath(file)
        try:
            with open(file_name, 'rb') as metadata_file:
                metadata_local = pickle.load(metadata_file)
                look_up_table.extend(metadata_local['look_up_table'])
                flows_information = metadata_local['flows_information']
                if flows_info is None:
                    flows_info = flows_information
                else:
                    flows_info['min_vx'] = min(flows_info['min_vx'], flows_information['min_vx'])
                    flows_info['min_vx'] = min(flows_info['min_vx'], flows_information['min_vy'])
                    flows_info['min_vz'] = min(flows_info['min_vz'], flows_information['min_vz'])
                    flows_info['max_vx'] = max(flows_info['max_vx'], flows_information['max_vx'])
                    flows_info['max_vy'] = max(flows_info['max_vy'], flows_information['max_vy'])
                    flows_info['max_vz'] = max(flows_info['max_vz'], flows_information['max_vz'])

        except FileNotFoundError:
            raise FileNotFoundError(
                "Metadata not found when merging individual metadata")

    # Save metadata into disk
    metadata = {'look_up_table': look_up_table,
                'flows_information': flows_info}
    with open(os.path.join(input_path, "metadata"), 'wb') as metadata_file:
        pickle.dump(metadata, metadata_file)


def _pad_batch(batch):
    # Get the number of points in the largest point cloud
    true_number_of_points = [e[1].shape[0] for e in batch]
    max_points_prev = np.max(true_number_of_points)
    point_features = batch[0][0].shape[1]

    # We need a mask of all the points that actually exist
    zeros = np.zeros((len(batch), max_points_prev), dtype=bool)
    # Mark all points that ARE NOT padded
    for i, n in enumerate(true_number_of_points):
        zeros[i, :n] = 1

    # resize all tensors to the max points size
    return [
        [
            np.resize(entry[0], (max_points_prev, point_features)),
            np.resize(entry[1], max_points_prev),
            zeros[i]
        ] for i, entry in enumerate(batch)
    ]


def _pad_targets(batch):
    true_number_of_points = [e.shape[0] for e in batch]
    max_points = np.max(true_number_of_points)
    target_features = batch[0].shape[1]
    return [
        np.resize(entry, (max_points, target_features))
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
    #   2. (batch_size, max_n_points, 2) the 2D grid_indices to map to
    #   3. (batch_size, max_n_points) the 0-1 encoding if the element is padded
    # Batch previous for the previous frame
    # Batch current for the current frame

    # The targets consist of
    #   (batch_size, max_n_points, target_features). should by 4D x,y,z flow and class id

    return (batch_previous, batch_current), batch_targets
