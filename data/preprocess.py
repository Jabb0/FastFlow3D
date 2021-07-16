import glob
import os
import pickle

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2, dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


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
        flows = flows[flows[:, -1] != -1]
        if flows.size != 0:  # May all the flows are invalid
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