import numpy as np
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    """
    A random dataset that acts like it is WaymoDataset
    """

    # Transform to convert the getitem to tensor
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max,
                 drop_invalid_point_function,
                 point_cloud_transform,
                 min_number_points=50000, max_number_points=150000,
                 desired_length=1000):
        self.max_number_points = max_number_points
        self.min_number_points = min_number_points
        self.z_max = z_max
        self.z_min = z_min
        self.y_max = y_max
        self.y_min = y_min
        self.x_max = x_max
        self.x_min = x_min
        self._length = desired_length
        self._random_state = np.random.default_rng()

        self._drop_invalid_point_function = drop_invalid_point_function
        self._point_cloud_transform = point_cloud_transform

    def __len__(self) -> int:
        return self._length

    def draw_random_frame(self):
        # Random number of points
        number_of_points = self._random_state.integers(self.min_number_points, self.max_number_points)
        frame = np.zeros((number_of_points, 5))
        frame[:, 0] = self._random_state.uniform(self.x_min, self.x_max, size=number_of_points)
        frame[:, 1] = self._random_state.uniform(self.y_min, self.y_max, size=number_of_points)
        frame[:, 2] = self._random_state.uniform(self.z_min, self.z_max, size=number_of_points)
        frame[:, 3:5] = self._random_state.uniform(-5, 5, size=(number_of_points, 2))
        return frame

    def __getitem__(self, index):
        """
        Create a single pointcloud simulated
        :param index: 
        :return: (N_points, 5 features) with x,y,z being the coordinates and the
        last two features being the laser features
        """
        current_frame = self.draw_random_frame()
        previous_frame = self.draw_random_frame()

        current_flows = self._random_state.uniform(-20, 20, size=(current_frame.shape[0], 3))

        # Drop invalid points according to the method supplied
        current_frame, current_flows = self._drop_invalid_point_function(current_frame, current_flows)
        previous_frame, _ = self._drop_invalid_point_function(previous_frame, None)

        # Perform the pillarization of the point_cloud
        current = self._point_cloud_transform(current_frame)
        previous = self._point_cloud_transform(previous_frame)
        # This returns a tuple of augmented pointcloud and grid indices

        return (previous, current), current_flows
