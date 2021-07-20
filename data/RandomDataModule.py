from pathlib import Path
from typing import Optional, Union, List, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .RandomDataset import RandomDataset
from .util import ApplyPillarization, drop_points_function


class RandomDataModule(pl.LightningDataModule):
    """
    Data module to prepare and load the waymo dataset.
    Using a data module streamlines the data loading and preprocessing process.
    """

    def __init__(self, dataset_directory,
                 # These parameters are specific to the dataset
                 grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max,
                 batch_size: int = 32,
                 has_test=False,
                 num_workers=1):
        super(RandomDataModule, self).__init__()
        self._dataset_directory = Path(dataset_directory)
        self._batch_size = batch_size
        self._train_ = None
        self._val_ = None
        self._test_ = None
        # This is a transformation class that applies to pillarization
        self._pillarization_transform = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=x_min,
                                                           y_min=y_min, z_min=z_min, z_max=z_max)

        # This returns a function that removes points that should not be included in the pillarization.
        # It also removes the labels if given.
        self._drop_points_function = drop_points_function(x_min=x_min,
                                                          x_max=x_max, y_min=y_min, y_max=y_max,
                                                          z_min=z_min, z_max=z_max)

        self._has_test = has_test
        self._num_workers = num_workers

        # Only required for this dataset type
        self.z_max = z_max
        self.z_min = z_min
        self.y_max = y_max
        self.y_min = y_min
        self.x_max = x_max
        self.x_min = x_min

    def prepare_data(self) -> None:
        """
        Preprocessing of the data only called on 1 GPU.
        Download and process the datasets here. E.g., tokenization.
        Everything that is not random and only necessary once.
        This is used to download the dataset to a local storage for example.
            Later the dataset is then loaded by every worker in the setup() method.
        :return: None
        """
        # No need to download stuff
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup of the datasets. Called on every GPU in distributed training.
        Do splits and build model internals here.
        :param stage: either 'fit', 'validate', 'test' or 'predict'
        :return: None
        """
        # The Dataset will apply a transformation to each pointcloud
        # This transformation consists of a pillarization

        self._train_ = RandomDataset(x_max=self.x_max, x_min=self.x_min, y_max=self.y_max, y_min=self.y_min,
                                     z_max=self.z_max, z_min=self.z_min,
                                     # This part is actually necessary to prepare the data
                                     point_cloud_transform=self._pillarization_transform,
                                     drop_invalid_point_function=self._drop_points_function)
        self._val_ = RandomDataset(x_max=self.x_max, x_min=self.x_min, y_max=self.y_max, y_min=self.y_min,
                                   z_max=self.z_max, z_min=self.z_min,
                                   # This part is actually necessary to prepare the data
                                   point_cloud_transform=self._pillarization_transform,
                                   drop_invalid_point_function=self._drop_points_function)
        if self._has_test:
            self._test_ = RandomDataset(x_max=self.x_max, x_min=self.x_min, y_max=self.y_max, y_min=self.y_min,
                                        z_max=self.z_max, z_min=self.z_min,
                                        # This part is actually necessary to prepare the data
                                        point_cloud_transform=self._pillarization_transform,
                                        drop_invalid_point_function=self._drop_points_function)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for training
        :return: the dataloader to use
        """
        return DataLoader(self._train_, self._batch_size, num_workers=self._num_workers,
                          collate_fn=custom_collate)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for validation
        :return: the dataloader to use
        """
        return DataLoader(self._val_, self._batch_size, shuffle=False, num_workers=self._num_workers,
                          collate_fn=custom_collate)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for testing
        :return: the dataloader to use
        """
        if not self._has_test:
            raise RuntimeError("No test dataset specified. Maybe set has_test=True in DataModule init.")
        return DataLoader(self._test_, self._batch_size, shuffle=False, num_workers=self._num_workers,
                          collate_fn=custom_collate)
