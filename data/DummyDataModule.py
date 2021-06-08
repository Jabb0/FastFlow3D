import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from typing import Optional, Union, List, Dict


class DummyDataModule(pl.LightningDataModule):
    """
    A dummy data module to show how to use the PyTorch Lightning Data Module.
    Using a data module streamlines the data loading and preprocessing process.
    """
    def __init__(self, dataset_directory, batch_size: int = 32, val_fraction=0.8):
        super(DummyDataModule, self).__init__()
        self._dataset_directory = dataset_directory
        self._batch_size = batch_size
        self._val_fraction = val_fraction
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self) -> None:
        """
        Preprocessing of the data only called on 1 GPU.
        Download and process the datasets here. E.g., tokenization.
        Everything that is not random and only necessary once.
        This is used to download the dataset to a local storage for example.
            Later the dataset is then loaded by every worker in the setup() method.
        :return: None
        """
        # Download the MNIST dataset using the torchvision dataset
        _ = torchvision.datasets.MNIST(root=self._dataset_directory, download=True, train=True)
        # Download the test dataset?
        _ = torchvision.datasets.MNIST(root=self._dataset_directory, download=True, train=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup of the datasets. Called on every GPU in distributed training.
        Do splits and build model internals here.
        :param stage: either 'fit', 'validate', 'test' or 'predict'
        :return: None
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        fashion_mnist_train_val = torchvision.datasets.MNIST(root=self._dataset_directory,
                                                             download=False, train=True, transform=transform)
        self.test = torchvision.datasets.MNIST(root=self._dataset_directory,
                                               download=False, train=False, transform=transform)
        train_val_len = len(fashion_mnist_train_val)
        val_size = int(train_val_len * self._val_fraction)
        train_size = train_val_len - val_size
        self.train, self.val = random_split(fashion_mnist_train_val, [train_size, val_size])

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for training
        :return: the dataloader to use
        """
        return DataLoader(self.train, self._batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for validation
        :return: the dataloader to use
        """
        return DataLoader(self.val, self._batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for testing
        :return: the dataloader to use
        """
        return DataLoader(self.test, self._batch_size)
