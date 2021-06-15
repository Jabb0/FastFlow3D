import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pathlib import Path

from typing import Optional, Union, List, Dict

from .WaymoDataset import WaymoDataset
from utils.pillars import create_pillars_matrix  # Python relative imports are interesting


class ApplyPillarization:
    def __init__(self, grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max):
        self._grid_cell_size = grid_cell_size
        self._z_max = z_max
        self._z_min = z_min
        self._y_max = y_max
        self._y_min = y_min
        self._x_max = x_max
        self._x_min = x_min

    """ Transforms an point cloud to the augmented pointcloud depending on Pillarization """
    def __call__(self, x):
        point_cloud, labels = x
        point_cloud, grid_indices, labels = create_pillars_matrix(point_cloud, labels,
                                                                  grid_cell_size=self._grid_cell_size,
                                                                  x_min=self._x_min, x_max=self._x_max,
                                                                  y_min=self._y_min,  y_max=self._y_max,
                                                                  z_min=self._z_min, z_max=self._z_max)
        return [point_cloud, grid_indices], labels


class WaymoDataModule(pl.LightningDataModule):
    """
    Data module to prepare and load the waymo dataset.
    Using a data module streamlines the data loading and preprocessing process.
    """
    def __init__(self, dataset_directory,
                 # These parameters are specific to the dataset
                 grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max,
                 batch_size: int = 32):
        super(WaymoDataModule, self).__init__()
        self._dataset_directory = Path(dataset_directory)
        self._batch_size = batch_size
        self._train_ = None
        self._val_ = None
        self._test_ = None
        self._pillarization_transform = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=x_min,
                                                           x_max=x_max, y_min=y_min, y_max=y_max,
                                                           z_min=z_min, z_max=z_max)

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
        # This transformation consists of a pillarization and the toTensor operation.
        transformations = transforms.Compose([
            self._pillarization_transform,
            transforms.ToTensor()
        ])

        self._train_ = WaymoDataset(self._dataset_directory.joinpath("train"), transform=transformations)
        self._val_ = WaymoDataset(self._dataset_directory.joinpath("valid"), transform=transformations)
        self._test_ = WaymoDataset(self._dataset_directory.joinpath("test"), transform=transformations)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for training
        :return: the dataloader to use
        """
        return DataLoader(self._train_, self._batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for validation
        :return: the dataloader to use
        """
        return DataLoader(self._val_, self._batch_size, shuffle=False)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for testing
        :return: the dataloader to use
        """
        return DataLoader(self._test_, self._batch_size, shuffle=False)
