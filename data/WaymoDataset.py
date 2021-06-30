from torch.utils.data import Dataset
import os
import numpy as np
import pickle

from data.util import get_coordinates_and_features


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
                 point_cloud_transform=None):
        """
        Args:
            data_path (string): Folder with the compressed data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        # Config parameters
        look_up_table_path = os.path.join(data_path, 'look_up_table')
        # It has information regarding the files and transformations

        self.data_path = data_path

        self._drop_invalid_point_function = drop_invalid_point_function
        self._point_cloud_transform = point_cloud_transform

        try:
            with open(look_up_table_path, 'rb') as look_up_table_file:
                self.look_up_table = pickle.load(look_up_table_file)
        except FileNotFoundError:
            raise FileNotFoundError("Look-up table not found, please create it by running preprocess.py")

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
        previous_frame = get_coordinates_and_features(previous_frame, transform=C_T_P)
        current_frame = get_coordinates_and_features(current_frame, transform=None)

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

    # TODO save into disk but careful and advise that we load from disk
    def get_flow_ranges(self):
        min_vx_global, max_vx_global = np.inf, -np.inf
        min_vy_global, max_vy_global = np.inf, -np.inf
        min_vz_global, max_vz_global = np.inf, -np.inf
        for i in range(0, len(self)):
            (previous_frame, current_frame), flows = self[i]
            min_vx, min_vy, min_vz = flows[:,:-1].min(axis=0)
            max_vx, max_vy, max_vz = flows[:,:-1].max(axis=0)
            min_vx_global = min(min_vx_global, min_vx)
            min_vy_global = min(min_vy_global, min_vy)
            min_vz_global = min(min_vz_global, min_vz)
            max_vx_global = max(max_vx_global, max_vx)
            max_vy_global = max(max_vy_global, max_vy)
            max_vz_global = max(max_vz_global, max_vz)
            print(f"{i} of {len(self)}")

        #return min_vx_global, max_vx_global, min_vy_global, max_vy_global, min_vz_global, max_vz_global
        return np.array([min_vx_global, min_vy_global, min_vz_global]), np.array([max_vx_global, max_vy_global, max_vz_global])

    def set_drop_invalid_point_function(self, drop_invalid_point_function):
        self._drop_invalid_point_function = drop_invalid_point_function

    def set_point_cloud_transform(self, point_cloud_transform):
        self._point_cloud_transform = point_cloud_transform

    def read_point_cloud_pair(self, index):
        """
        Read from disk the current and prvious point cloud given an index
        """
        current_frame = np.load(os.path.join(self.data_path, self.look_up_table[index][0][0]))
        previous_frame = np.load(os.path.join(self.data_path, self.look_up_table[index][1][0]))
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
