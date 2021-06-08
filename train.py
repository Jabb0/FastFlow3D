import torch
from argparse import ArgumentParser
import pytorch_lightning as pl

from pathlib import Path

from data.DummyDataModule import DummyDataModule
from models.DummyModel import DummyModel


def cli():
    parser = ArgumentParser()
    parser.add_argument('data_directory', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--val_fraction', default=0.8, type=float)
    parser = pl.Trainer.add_argparse_args(parser)  # Add arguments for the trainer
    # Add model specific arguments here
    parser = DummyModel.add_model_specific_args(parser)
    args = parser.parse_args()

    dataset_path = Path(args.data_directory)
    # Create the dataset path
    dataset_path.mkdir(parents=True, exist_ok=True)

    model = DummyModel(args.hidden_dim, args.learning_rate)
    mnist_data_module = DummyDataModule(dataset_path, batch_size=args.batch_size, val_fraction=args.val_fraction)

    trainer = pl.Trainer.from_argparse_args(args,
                                            progress_bar_refresh_rate=25,  # Prevents Google Colab crashes
                                            gpus=1 if torch.cuda.is_available() else 0,
                                            )  # Add Trainer hparams if desired
    # The actual train loop
    trainer.fit(model, mnist_data_module)

    # Run also the testing
    trainer.test()  # Also loads the best checkpoint automatically


if __name__ == '__main__':
    cli()

