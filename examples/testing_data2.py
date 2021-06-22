import os
import tensorflow as tf
import math
import numpy as np
import itertools
import open3d as o3d
#tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2

FILENAME = '../data/train_tfrecord/train_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord'

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
counter = 0
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    counter += 1
    if counter > 100:
        break


def parse_range_image_and_camera_projection2(frame):
  """Parse range images and camera projections given a frame.

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


(range_images, camera_projections, point_flows,
 range_image_top_pose) = parse_range_image_and_camera_projection2(
    frame)


def convert_range_image_to_point_cloud2(frame,
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


def visualize_point_cloud(points):
    """ Input must be a point cloud of shape (n_points, 3) """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])


points, cp_points, flows = convert_range_image_to_point_cloud2(
    frame,
    range_images,
    camera_projections,
    point_flows,
    range_image_top_pose)

kk = 0

points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose,
    ri_index=1)

# 3d points in vehicle frame.
points_all = np.concatenate(points, axis=0)
flows_all = np.concatenate(flows, axis=0)
points_all_ri2 = np.concatenate(points_ri2, axis=0)
# camera projection corresponding to each point.
cp_points_all = np.concatenate(cp_points, axis=0)
cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)


print(points_all.shape) # I guess they are in the AV reference frame
print(points_all[0, :])
visualize_point_cloud(points_all)
print(cp_points_all.shape)
print(points_all[0:2])
print(cp_points_all[5000:5003])
print(flows_all[5000:5003])

for i in range(0, len(flows_all)):
    if flows_all[i][0] != 0:
        print(flows_all[i])

for i in range(5): # Because of 5 lidars sensors
  print(points[i].shape)
  print(cp_points[i].shape)
  print(flows[i].shape)

