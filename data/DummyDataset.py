from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class DummyDataset(Dataset):
    """
    Simple dummy dataset.
    All integers from 1 to a given limit dividable by a given divisor
    """

    def __init__(self, divisor, limit,) -> None:
        super().__init__()
        self.data = [i for i in range(1, limit + 1) if i % divisor == 0]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index]
