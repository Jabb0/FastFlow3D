from pathlib import Path
from typing import Optional, Union, List, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .FlyingThings3DDataset import FlyingThings3DDataset


class FlyingThings3DDataModule(pl.LightningDataModule):
    """
    Data module to prepare and load the waymo dataset.
    Using a data module streamlines the data loading and preprocessing process.
    """

    def __init__(self, dataset_directory,
                 batch_size: int = 32,
                 has_test=False,
                 num_workers=1,
                 n_points=None):
        super(FlyingThings3DDataModule, self).__init__()
        self._dataset_directory = Path(dataset_directory)
        self._batch_size = batch_size
        self._train_ = None
        self._val_ = None
        self._test_ = None

        self._has_test = has_test
        self._num_workers = num_workers

        self._n_points = n_points

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
        self._train_ = FlyingThings3DDataset(self._dataset_directory.joinpath("train"),
                                             n_points=self._n_points)
        self._val_ = FlyingThings3DDataset(self._dataset_directory.joinpath("valid"),
                                           n_points=self._n_points)
        if self._has_test:
            self._test_ = FlyingThings3DDataset(self._dataset_directory.joinpath("test"),
                                                n_points=self._n_points)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for training
        :return: the dataloader to use
        """
        return DataLoader(self._train_, self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for validation
        :return: the dataloader to use
        """
        return DataLoader(self._val_, self._batch_size, shuffle=False, num_workers=self._num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for testing
        :return: the dataloader to use
        """
        if not self._has_test:
            raise RuntimeError("No test dataset specified. Maybe set has_test=True in DataModule init.")
        return DataLoader(self._test_, self._batch_size, shuffle=False, num_workers=self._num_workers)
