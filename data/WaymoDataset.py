from torch.utils.data import Dataset
import tensorflow as tf
import os
from waymo_open_dataset import dataset_pb2 as open_dataset
from data.util import convert_range_image_to_point_cloud, parse_range_image_and_camera_projection
import numpy as np
import pickle

# TODO: tensor operations to make it faster?
# TODO: look up table and prepressing
# TODO: check context name to ensure two consecutive frames
class WaymoDataset(Dataset):
    """
    Waymo Custom Dataset for flow estimation. For a detailed description of each
    field please refer to:
    https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
    """

    def save_point_cloud(self, compressed_frame, file_path):
        frame = self.get_uncompressed_frame(compressed_frame)
        points, flows = self.compute_features(frame)
        point_cloud = np.hstack((points, flows))
        np.save(file_path, point_cloud)
        transform = list(frame.pose.transform)
        return points, flows, transform

    def preprocess(self, tfrecord_path, output_path):
        # Look is a list of lists of tuples:
        # [[t_1, t_0], [t_2, t_1], ... , [t_n, t_(n-1)]]
        # where t_i is the file_path
        look_up_table = []
        data_files = os.listdir(tfrecord_path)
        for i, data_file in enumerate(data_files):
            data_file_path = os.path.join(tfrecord_path, data_file)
            loaded_file = tf.data.TFRecordDataset(data_file_path, compression_type='')
            previous_frame = None
            for j, frame in enumerate(loaded_file):
                point_cloud_path = os.path.join(output_path, "pointCloud_file_" + str(i) + "_frame_" + str(j) + ".npy")
                # Process frame and store point clouds into disk
                _, _, pose_transform = self.save_point_cloud(frame, point_cloud_path)
                if j == 0:
                    previous_frame = (point_cloud_path, pose_transform)
                else:
                    current_frame = (point_cloud_path, pose_transform)
                    look_up_table.append([current_frame, previous_frame])
                    previous_frame = current_frame
                if j==5:
                    break
        return look_up_table

    # Transform to convert the getitem to tensor
    def __init__(self, data_path, transform=None, force_preprocess=False, tfrecord_path=None):
        """
        Args:
            data_path (string): Folder with the compressed data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        # Config parameters
        look_up_table_path = os.path.join(data_path,'look_up_table')  # It has information regarding the files and transformations

        if force_preprocess:
            if tfrecord_path is None:
                raise ValueError("tfrecord_path cannot be None when forcing preprocess")
            else:
                look_up_table = self.preprocess(tfrecord_path, data_path)
                with open(look_up_table_path, 'wb') as look_up_table_file:
                    pickle.dump(look_up_table, look_up_table_file)

        try:
            with open(look_up_table_path, 'rb') as look_up_table_file:
                self.look_up_table = pickle.load(look_up_table_file)
        except FileNotFoundError:
            raise FileNotFoundError("Look-up table not found, please create it by running the file with force_preprocess=True")

        self.data_path = data_path

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

    def compute_features(self, frame):
        """
        :param frame: Uncompressed frame
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
        points_all = np.hstack((points_coord, points_features))
        return points_all, flows_all

    def get_coordinates_and_features(self, point_cloud, transform=None):
        # :param transform: Optional, transformation matrix to apply
        points_coord, features, flows = point_cloud[:, 0:3], point_cloud[:, 3:5], point_cloud[:, 5:]
        if transform is not None:
            ones = np.ones((points_coord.shape[0], 1))
            points_coord = np.hstack((points_coord, ones))
            points_coord = transform @ points_coord.T
            points_coord = points_coord[0:-1, :]
            points_coord = points_coord.T
        point_cloud = np.hstack((points_coord, features))
        return point_cloud

    def read_point_cloud_pair(self, index):
        current_frame = np.load(self.look_up_table[index][0][0])
        previous_frame = np.load(self.look_up_table[index][1][0])
        return current_frame, previous_frame

    def get_pose_transform(self, index):
        current_frame_pose = self.look_up_table[index][0][1]
        previous_frame_pose = self.look_up_table[index][1][1]
        return current_frame_pose, previous_frame_pose

    def get_flows(self, frame):
        flows = frame[:, -4:]
        return flows

    def __getitem__(self, index):
        """
        Return two point clouds, the current point and its previous one. It also
        return the flow per each point of the current cloud

        A point cloud has a shape of [N, 3], being N the number of points and the
        second dimensions corresponds to [x, y, z], where (x, y, z) is the point position
        in the current frame.

        """
        current_frame, previous_frame = self.read_point_cloud_pair(index)
        current_frame_pose, previous_frame_pose = self.get_pose_transform(index)
        flows = self.get_flows(current_frame)

        # G_T_C -> Global_TransformMatrix_Current
        G_T_C = np.reshape(np.array(current_frame_pose), [4, 4])

        # G_T_P -> Global_TransformMatrix_Previous
        G_T_P = np.reshape(np.array(previous_frame_pose), [4, 4])
        C_T_P = np.linalg.inv(G_T_C) @ G_T_P
        previous_frame = self.get_coordinates_and_features(previous_frame, transform=C_T_P)
        current_frame = self.get_coordinates_and_features(current_frame, transform=None)

        return [current_frame, previous_frame], flows
