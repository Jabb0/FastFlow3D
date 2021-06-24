from torch.utils.data import Dataset
import tensorflow as tf
import os
from waymo_open_dataset import dataset_pb2 as open_dataset
from data.util import convert_range_image_to_point_cloud, parse_range_image_and_camera_projection
import numpy as np
import pickle

# TODO: NOW IS HARDCODED TO STOP IN THE FIFTH FRAME (for debugging purposes)
# TODO: tensor operations to make it faster?
# TODO: check context name to ensure two consecutive frames
class WaymoDataset(Dataset):
    """
    Waymo Custom Dataset for flow estimation. For a detailed description of each
    field please refer to:
    https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
    """


    def __init__(self, data_path,
                 drop_invalid_point_function=None,
                 point_cloud_transform=None,
                 force_preprocess=False, tfrecord_path=None,
                 frames_per_segment=None):
        """
        Args:
            data_path (string): Folder with the compressed data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            force_preprocess (bool, optional): If true, apply preprocessing storing point clouds
            into disk
            tfrecord_path (string, optional): path to store the processes point cloud.
            Only needed if force_preprocess=True
        """
        super().__init__()
        # Config parameters
        look_up_table_path = os.path.join(data_path, 'look_up_table')
        # It has information regarding the files and transformations

        self.data_path = data_path

        self._drop_invalid_point_function = drop_invalid_point_function
        self._point_cloud_transform = point_cloud_transform

        self._frames_per_segment = frames_per_segment

        if force_preprocess:
            if tfrecord_path is None:
                raise ValueError("tfrecord_path cannot be None when forcing preprocess")
            else:
                # Save look-up-table into disk
                look_up_table = self.preprocess(tfrecord_path, data_path)
                with open(look_up_table_path, 'wb') as look_up_table_file:
                    pickle.dump(look_up_table, look_up_table_file)

        try:
            with open(look_up_table_path, 'rb') as look_up_table_file:
                self.look_up_table = pickle.load(look_up_table_file)
        except FileNotFoundError:
            raise FileNotFoundError("Look-up table not found, please create it by running the file with force_preprocess=True")

    def __len__(self) -> int:
        return len(self.look_up_table)

    def __getitem__(self, index):
        """
        Return two point clouds, the current point and its previous one. It also
        return the flow per each point of the current cloud

        A point cloud has a shape of [N, F], being N the number of points and the
        F to the number of features, which is [x, y, z, intensity, elongation]
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

        # Drop invalid points according to the method supplied
        if self._drop_invalid_point_function is not None:
            current_frame, flows = self._drop_invalid_point_function(current_frame, flows)
            previous_frame, _ = self._drop_invalid_point_function(previous_frame, None)

        # Perform the pillarization of the point_cloud
        if self._point_cloud_transform is not None:
            current_frame = self._point_cloud_transform(current_frame)
            previous_frame = self._point_cloud_transform(previous_frame)
        # This returns a tuple of augmented pointcloud and grid indices

        return (previous_frame, current_frame), flows

    def save_point_cloud(self, compressed_frame, file_path):
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
        frame = self.get_uncompressed_frame(compressed_frame)
        points, flows = self.compute_features(frame)
        point_cloud = np.hstack((points, flows))
        np.save(file_path, point_cloud)
        transform = list(frame.pose.transform)
        return points, flows, transform

    def preprocess(self, tfrecord_path, output_path):
        """
        Preprocess the TFRecord data to store in a suitable form for training
        in disk. A point cloud in disk has dimensions [N, 9] where N is the number of points
        and per each point it stores [x, y, z, intensity, elongation, vx, vy, vz, label]
        :param tfrecord_path: path where are the TFRecord files. They should have the flow extension.
                              They can be downloaded from https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow
        :param output_path: path where the processed point clouds will be saved.
        :return: look_up_table. It has the form [[t_1, t_0], [t_2, t_1], ... , [t_n, t_(n-1)]], where t_i is
                                (file_path, transform), where file_path is the file where the point cloud is stored
                                and transform the transformation to apply to a point to change it reference frame from global
                                to the car frame in that moment.
        """
        look_up_table = []
        data_files = os.listdir(tfrecord_path)
        for i, data_file in enumerate(data_files):
            print(f"Processing file {i + 1} out of {len(data_files)} ({(i + 1)/len(data_files):.2f}%)")
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
                if self._frames_per_segment is not None and j == self._frames_per_segment:  # TODO remove this in the final version
                    break
        return look_up_table

    def get_uncompressed_frame(self, compressed_frame):
        """
        :param compressed_frame: Compressed frame
        :return: Uncompressed frame
        """
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(compressed_frame.numpy()))
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
        # Note that first are features and then point coordinates
        points_features, points_coord = points_all[:, 1:3], points_all[:, 3:points_all.shape[1]]
        points_all = np.hstack((points_coord, points_features))
        return points_all, flows_all

    def get_coordinates_and_features(self, point_cloud, transform=None):
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

    def read_point_cloud_pair(self, index):
        """
        Read from disk the current and prvious point cloud given an index
        """
        current_frame = np.load(self.look_up_table[index][0][0])
        previous_frame = np.load(self.look_up_table[index][1][0])
        return current_frame, previous_frame

    def get_pose_transform(self, index):
        """
        Return the frame poses of the current and previous point clouds given an index
        """
        current_frame_pose = self.look_up_table[index][0][1]
        previous_frame_pose = self.look_up_table[index][1][1]
        return current_frame_pose, previous_frame_pose

    def get_flows(self, frame):
        """
        Return the flows given a point cloud
        """
        flows = frame[:, -4:]
        return flows
