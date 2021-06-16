import tensorflow as tf
import time

from utils.plot import visualize_point_cloud, plot_pillars, plot_2d_point_cloud
from data.util import convert_range_image_to_point_cloud, parse_range_image_and_camera_projection
from utils.pillars import create_pillars_matrix
from networks.encoder import PillarFeatureNet
from data.WaymoDataset import WaymoDataset
import os

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



if __name__ == '__main__':
    disable_gpu()  # FIXME cannot execute the code without disabling GPU

    # Getting points with the dataloader
    train_path = 'data/train'
    tfrecord_path = 'data/train_tfrecord'
    preprocess = False
    t = time.time()
    waymo_dataset = WaymoDataset(train_path, force_preprocess=preprocess, tfrecord_path=tfrecord_path)
    print(f"Preprocessing duration: {(time.time() - t):.2f} s")

    accum = 0
    t = time.time()
    for i, item in enumerate(waymo_dataset):
        accum += (time.time() - t)
        t = time.time()
    print(f"Mean access time: {(accum/len(waymo_dataset)):.2f} s")


