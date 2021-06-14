import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pathlib import Path

from typing import Optional, Union, List, Dict

from .WaymoDataset import WaymoDataset


class WaymoDataModule(pl.LightningDataModule):
    """
    Data module to prepare and load the waymo dataset.
    Using a data module streamlines the data loading and preprocessing process.
    """
    def __init__(self, dataset_directory, batch_size: int = 32):
        super(WaymoDataModule, self).__init__()
        self._dataset_directory = Path(dataset_directory)
        self._batch_size = batch_size
        self._train_ = None
        self._val_ = None
        self._test_ = None

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
        self._train_ = WaymoDataset(self._dataset_directory.joinpath("train"))
        self._val_ = WaymoDataset(self._dataset_directory.joinpath("valid"))
        self._test_ = WaymoDataset(self._dataset_directory.joinpath("test"))

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
