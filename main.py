import os
import time

import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

from data.WaymoDataset import WaymoDataset
from data.preprocess import convert_range_image_to_point_cloud, parse_range_image_and_camera_projection
from networks.encoder import PillarFeatureNet
from utils.pillars import create_pillars_matrix
from utils.plot import visualize_point_cloud


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

    # Getting points with the dataloader
    train_path = './data/train'
    #train_path = '/mnt/LinuxGames/deeplearninglab/dataset/waymo_flow/train'
    arr = os.listdir(train_path)
    waymo_dataset = WaymoDataset(train_path)

    [points_current_frame, points_previous_frame], flows = waymo_dataset[0]
    points_all = points_current_frame

    print(points_all.shape)  # I guess they are in the AV reference frame
    points_coord = points_all[:, 0:3]
    visualize_point_cloud(points_coord)

    # Pillar transformation
    grid_cell_size = 0.16

    x_max = 81.92
    x_min = 0

    y_max = 40.96
    y_min = -40.96

    z_max = 3
    z_min = -3

    t = time.time()
    points, indices, flows = create_pillars_matrix(points_all, flows, grid_cell_size=grid_cell_size, x_min=x_min,
                                                   x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)
    print(f"Pillar transformation duration: {(time.time() - t):.2f} s")
    # plot_pillars(indices=indices, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, grid_cell_size=grid_cell_size)
    # plot_2d_point_cloud(pc=points_all)

    import torch
    t = time.time()
    # unsqueeze and repeat to just add a batch dim
    points = torch.tensor(points, dtype=torch.float32).unsqueeze(0).repeat(5, 1, 1)
    indices = torch.tensor(indices, dtype=torch.float32).unsqueeze(0).repeat(5, 1, 1)

    # ------- PillarFeatureNet PART ------- #
    pfn = PillarFeatureNet(n_pillars_x=512, n_pillars_y=512)
    embedded_points, grid = pfn(points, indices)
    print(f"PillarFeatureNet duration: {(time.time() - t):.2f} s")

    # ------- UNet PART ------- #
    from networks.convEncoder import ConvEncoder
    from networks.convDecoder import ConvDecoder
    unet_encoder = ConvEncoder()
    unet_decoder = ConvDecoder()

    # grid: Output of the PillarFeatureNet!!! (also called B in paper)
    # batch_size x embedding_size x grid_size x grid_size
    # Later: This is should be the grid of the encoder output
    grid_prev = torch.rand(1, 64, 512, 512)  # also called B
    grid_cur = torch.rand(1, 64, 512, 512)  # also called B

    print("\nEncoder Output:")
    F_prev, L_prev, R_prev = unet_encoder(grid_prev)
    F_cur, L_cur, R_cur = unet_encoder(grid_cur)
    print("\nDecoder Output:")
    decoder_output = unet_decoder(B_prev=grid_prev, F_prev=F_prev, L_prev=L_prev, R_prev=R_prev,
                                  B_cur=grid_cur, F_cur=F_cur, L_cur=L_cur, R_cur=R_cur)



