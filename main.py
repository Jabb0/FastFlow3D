import numpy as np
import tensorflow as tf

from utils.plot import visualize_point_cloud, plot_pillars, plot_2d_point_cloud
from data.util import convert_range_image_to_point_cloud, parse_range_image_and_camera_projection
from utils.pillars import create_pillars, assign_points_to_pillars

from waymo_open_dataset import dataset_pb2 as open_dataset


def disable_gpu():
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def read_data(fname):
    dataset = tf.data.TFRecordDataset(fname, compression_type='')
    counter = 0
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        counter += 1
        if counter > 100:
            break

    range_images, camera_projections, point_flows, range_image_top_pose = parse_range_image_and_camera_projection(frame)

    points, cp_points, flows = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        point_flows,
        range_image_top_pose)

    return points, cp_points, flows


if __name__ == '__main__':
    disable_gpu()  # FIXME cannot execute the code without disabling GPU

    fname = 'data/train/train_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord'
    points, cp_points, flows = read_data(fname=fname)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    flows_all = np.concatenate(flows, axis=0)

    print(points_all.shape)  # I guess they are in the AV reference frame
    #visualize_point_cloud(points_all)

    # Pillar transformation
    cp = points_all
    grid_size = 10
    pillars = create_pillars(cp, grid_size=grid_size)
    assign_points_to_pillars(cp, pillars)
    plot_pillars(cp=cp, pillars=pillars, grid_size=grid_size)
    plot_2d_point_cloud(cp=cp)


