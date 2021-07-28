import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


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
        return len(glob.glob(os.path.join(self.data_path, '*.npz')))

    def __getitem__(self, index):
        """
        Return two point clouds, the current point and its previous one. It also
        return the flow per each point of the current cloud

        A point cloud has a shape of [N, F], being N the number of points and the
        F to the number of features, which is [x, y, z, intensity, elongation]
        """
        current_frame, previous_frame, c2, c1, flows, mask = self.read_point_cloud_pair_and_flow(index)

        if self._n_points is not None:
            current_frame, previous_frame, c2, c1, flows, mask = self.subsample_points(
                current_frame, previous_frame, c2, c1, flows, mask)

        previous_frame = (torch.as_tensor(previous_frame), torch.as_tensor(c1))
        current_frame = (torch.as_tensor(current_frame), torch.as_tensor(c2), mask)
        return (previous_frame, current_frame), flows

    def subsample_points(self, current_frame, previous_frame, c2, c1, flows, mask):
        # current_frame.shape[0] == flows.shape[0]
        if current_frame.shape[0] > self._n_points:
            indexes_current_frame = np.linspace(0, current_frame.shape[0]-1, num=self._n_points).astype(int)
            current_frame = current_frame[indexes_current_frame, :]
            flows = flows[indexes_current_frame, :]
            mask = mask[indexes_current_frame]
            c2 = c2[indexes_current_frame, :]
        if previous_frame.shape[0] > self._n_points:
            indexes_previous_frame = np.linspace(0, previous_frame.shape[0]-1, num=self._n_points).astype(int)
            previous_frame = previous_frame[indexes_previous_frame, :]
            c1 = c1[indexes_previous_frame, :]
        return current_frame, previous_frame, c1, c2, flows, mask

    def set_drop_invalid_point_function(self, drop_invalid_point_function):
        self._drop_invalid_point_function = drop_invalid_point_function

    def read_point_cloud_pair_and_flow(self, index):
        """
        Read from disk the frame of the given an index
        """
        frame_list = glob.glob(os.path.join(self.data_path, '*.npz'))
        frame_fname = frame_list[index]
        frame = np.load(frame_fname)

        pos1 = frame['points1'].astype('float32')
        pos2 = frame['points2'].astype('float32')
        color1 = frame['color1'].astype('float32')
        color2 = frame['color2'].astype('float32')
        flow = frame['flow'].astype('float32')
        mask1 = frame['mask']

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos2, pos1, color2, color1, flow, mask1
