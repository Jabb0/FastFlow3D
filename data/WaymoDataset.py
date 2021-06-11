from torch.utils.data import Dataset
import tensorflow as tf
import os
from waymo_open_dataset import dataset_pb2 as open_dataset
from data.util import convert_range_image_to_point_cloud, parse_range_image_and_camera_projection
from utils.pillars import create_pillars, assign_points_to_pillars
import numpy as np

class WaymoDataset(Dataset):

    # Transform to convert the getitem to tensor
    def __init__(self, data_path, pillars_grid=10, transform=None):
        #super().__init__()
        self.data_path = data_path
        self.transform = transform

        # Samples is a list of tuples, [(t_1, t_0), (t_2, t_1), ... , (t_n, t_(n-1))]
        self.compressed_samples = []
        self.grid_size = pillars_grid

        # Load into memory the dataset. Be careful since may it can run out of memory
        data_files = os.listdir(self.data_path)
        for data_file in data_files:
            data_file_path = os.path.join(self.data_path, data_file)
            loaded_file = tf.data.TFRecordDataset(data_file_path, compression_type='')
            previous_frame = None
            for count, frame in enumerate(loaded_file):
                if count == 0:
                    previous_frame = frame
                else:
                    self.compressed_samples.append((frame, previous_frame))
                    previous_frame = frame

        kk = 0

    def __len__(self) -> int:
        return len(self.compressed_samples)

    def get_uncompressed_frame(self, compressed_frame):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(compressed_frame.numpy()))
        return frame

    def compute_features(self, frame):
        range_images, camera_projections, point_flows, range_image_top_pose = parse_range_image_and_camera_projection(
            frame)

        points, cp_points, flows = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            point_flows,
            range_image_top_pose)

        points_all = np.concatenate(points, axis=0)
        flows_all = np.concatenate(flows, axis=0)

        pillars = create_pillars(points_all, grid_size=self.grid_size)
        assign_points_to_pillars(points_all, pillars)
        kk = 0

    # This method is exclusively for training so pillars should be applied
    # TODO convert coordinates from one frame to another
    def __getitem__(self, index):
        current_frame = self.compressed_samples[index][0]
        current_frame = self.get_uncompressed_frame(current_frame)
        previous_frame = self.compressed_samples[index][1]

        current_frame = self.compute_features(current_frame)

        return pillars
