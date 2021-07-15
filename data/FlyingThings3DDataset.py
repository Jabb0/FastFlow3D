from torch.utils.data import Dataset
import os
import numpy as np
import glob


class FlyingThings3DDataset(Dataset):
    """
    """
    def __init__(self, data_path, n_points,
                 drop_invalid_point_function=None):
        """
        Args:
            data_path (string): Folder with the compressed data.
        """
        super().__init__()
        self.data_path = data_path
        self._n_points = n_points
        self._drop_invalid_point_function = drop_invalid_point_function

    def __len__(self) -> int:
        print(len(glob.glob(os.path.join(self.data_path, '*.npz'))))
        return len(glob.glob(os.path.join(self.data_path, '*.npz')))

    def __getitem__(self, index):
        """
        Return two point clouds, the current point and its previous one. It also
        return the flow per each point of the current cloud

        A point cloud has a shape of [N, F], being N the number of points and the
        F to the number of features, which is [x, y, z, intensity, elongation]
        """
        current_frame, previous_frame, flows = self.read_point_cloud_pair_and_flow(index)

        if self._n_points is not None:
            current_frame, previous_frame, flows = self.subsample_points(current_frame, previous_frame, flows)

        # Drop invalid points according to the method supplied
        if self._drop_invalid_point_function is not None:
            current_frame, flows = self._drop_invalid_point_function(current_frame, flows)
            previous_frame, _ = self._drop_invalid_point_function(previous_frame, None)
        print(current_frame.shape)
        return (previous_frame, current_frame), flows

    def subsample_points(self, current_frame, previous_frame, flows):
        # current_frame.shape[0] == flows.shape[0]
        if current_frame.shape[0] > self._n_points:
            indexes_current_frame = np.linspace(0, current_frame.shape[0]-1, num=self._n_points).astype(int)
            current_frame = current_frame[indexes_current_frame, :]
            flows = flows[indexes_current_frame, :]
        if previous_frame.shape[0] > self._n_points:
            indexes_previous_frame = np.linspace(0, previous_frame.shape[0]-1, num=self._n_points).astype(int)
            previous_frame = previous_frame[indexes_previous_frame, :]
        return current_frame, previous_frame, flows

    def set_drop_invalid_point_function(self, drop_invalid_point_function):
        self._drop_invalid_point_function = drop_invalid_point_function

    def read_point_cloud_pair_and_flow(self, index):
        """
        Read from disk the frame of the given an index
        """
        frame = np.load(os.path.join(self.data_path, 'frame_{}.npz'.format(index)))
        return frame['points2'], frame['points1'], frame['flow']
