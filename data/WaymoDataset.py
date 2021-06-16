from torch.utils.data import Dataset
import tensorflow as tf
import os
from waymo_open_dataset import dataset_pb2 as open_dataset
from data.util import convert_range_image_to_point_cloud, parse_range_image_and_camera_projection
import numpy as np

# TODO: tensor operations to make it faster?
# TODO: look up table and prepressing
# TODO: check context name to ensure two consecutive frames
class WaymoDataset(Dataset):
    """
    Waymo Custom Dataset for flow estimation. For a detailed description of each
    field please refer to:
    https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
    """

    # Transform to convert the getitem to tensor
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Folder with the compressed data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform

        # Look is a list of lists of tuples:
        # [[t_1, t_0], [t_2, t_1], ... , [t_n, t_(n-1)]]
        # where t_i is ['file_path', indexInsideTheTFRECORD]
        self.look_up_table = []
        # Load into memory the dataset. Be careful since may it can run out of memory
        data_files = os.listdir(self.data_path)
        for data_file in data_files:
            data_file_path = os.path.join(self.data_path, data_file)
            loaded_file = tf.data.TFRecordDataset(data_file_path, compression_type='')
            previous_frame = None
            for count, _ in enumerate(loaded_file):
                if count == 0:
                    previous_frame = (data_file_path, count)
                else:
                    current_frame = (data_file_path, count)
                    self.look_up_table.append([current_frame, previous_frame])
                    previous_frame = current_frame

    def __len__(self) -> int:
        return len(self.look_up_table)

    def get_uncompressed_frame(self, compressed_frame):
        """
        :return: Uncompressed frame
        """
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(compressed_frame.numpy()))
        #print(frame.context.name)
        return frame

    def compute_features(self, frame, transform=None):
        """
        :param frame: Uncompressed frame
        :param transform: Optional, transformation matrix to apply
        :return: [N, F], [N, 4], where N is the number of points, F the number of features,
        which is [x, y, z, intensity, elongation] and 4 in the second results stands for [vx, vy, vz, label], which corresponds
        to the flow information
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
        points_coord, points_features = points_all[:, 0:3], points_all[:, 4:points_all.shape[1]]

        if transform is not None:
            ones = np.ones((points_coord.shape[0], 1))
            points_coord = np.hstack((points_coord, ones))
            points_coord = transform @ points_coord.T
            points_coord = points_coord[0:-1, :]
            points_coord = points_coord.T

        points_all = np.hstack((points_coord, points_features))
        return points_all, flows_all

    def read_frame(self, data_file_path, index):
        loaded_file = tf.data.TFRecordDataset(data_file_path, compression_type='')
        previous_frame = None
        for count, frame in enumerate(loaded_file):
            if count == index:
                return frame

    def read_frames_pair(self, index):
        current_frame = self.read_frame(self.look_up_table[index][0][0], self.look_up_table[index][0][1])
        previous_frame = self.read_frame(self.look_up_table[index][1][0], self.look_up_table[index][1][1])
        return current_frame, previous_frame

    def __getitem__(self, index):
        """
        Return two point clouds, the current point and its previous one. It also
        return the flow per each point of the current cloud

        A point cloud has a shape of [N, 3], being N the number of points and the
        second dimensions corresponds to [x, y, z], where (x, y, z) is the point position
        in the current frame.

        """
        current_frame, previous_frame = self.read_frames_pair(index)
        current_frame = self.get_uncompressed_frame(current_frame)

        # G_T_C -> Global_TransformMatrix_Current
        G_T_C = np.reshape(np.array(current_frame.pose.transform), [4, 4])
        current_frame, flows = self.compute_features(current_frame)

        previous_frame = self.get_uncompressed_frame(previous_frame)

        # G_T_P -> Global_TransformMatrix_Previous
        G_T_P = np.reshape(np.array(previous_frame.pose.transform), [4, 4])
        C_T_P = np.linalg.inv(G_T_C) @ G_T_P
        previous_frame, _ = self.compute_features(previous_frame, transform=C_T_P)

        return [current_frame, previous_frame], flows
