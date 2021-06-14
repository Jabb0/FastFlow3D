import numpy as np
import tensorflow as tf
import time

from utils.plot import visualize_point_cloud, plot_pillars, plot_2d_point_cloud
from data.util import convert_range_image_to_point_cloud, parse_range_image_and_camera_projection
from utils.pillars import create_pillars
from networks.encoder import PillarFeatureNet

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
        frame.ParseFromString(bytearray(data.numpy()))  # Uncompress frame
        counter += 1
        if counter > 100:  # 100 in order to not take the first frame in case there is no flow information
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
    # visualize_point_cloud(points_all)

    # Pillar transformation
    grid_cell_size = 0.16

    x_max = 81.92
    x_min = 0

    y_max = 40.96
    y_min = -40.96

    z_max = 3
    z_min = -3

    t = time.time()
    points, indices = create_pillars(points_all, grid_cell_size=grid_cell_size, x_min=x_min, x_max=x_max,
                                     y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)
    print(f"Pillar transformation duration: {(time.time() - t):.2f} s")
    # plot_pillars(indices=indices, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, grid_cell_size=grid_cell_size)
    # plot_2d_point_cloud(pc=points_all)

    import torch

    # pfn = PillarFeatureNet(x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, grid_cell_size=grid_cell_size)
    # output = pfn(torch.tensor(points, dtype=torch.float32), torch.tensor(indices, dtype=torch.float32))
    # print(points.shape)
    # print(indices.shape)
    # print(output.shape)

    from networks.convEncoder import ConvEncoder
    unet = ConvEncoder()
    # batch_size x embedding_size x grid_size x grid_size
    input = torch.rand(1, 64, 512, 512)

    output = unet(input)

    print(output.shape)



