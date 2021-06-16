import torch
from argparse import ArgumentParser
import pytorch_lightning as pl

from pathlib import Path

from data import WaymoDataModule, RandomDataModule
from models import FastFlow3DModel


def cli():
    parser = ArgumentParser()
    parser.add_argument('data_directory', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--x_max', default=81.92, type=float)
    parser.add_argument('--x_min', default=0, type=float)
    parser.add_argument('--y_max', default=40.96, type=float)
    parser.add_argument('--y_min', default=-40.96, type=float)
    parser.add_argument('--z_max', default=3, type=float)
    parser.add_argument('--z_min', default=-3, type=float)
    parser.add_argument('--grid_cell_size', default=0.16, type=float)
    parser.add_argument('--test_data_available', default=False, type=bool)
    parser.add_argument('--fast_dev_run', default=True, type=bool)
    parser.add_argument('--num_workers', default=1, type=int)

    # NOTE: Readd this to see all parameters of the trainer
    # parser = pl.Trainer.add_argparse_args(parser)  # Add arguments for the trainer
    # Add model specific arguments here
    parser = FastFlow3DModel.add_model_specific_args(parser)
    args = parser.parse_args()

    dataset_path = Path(args.data_directory)
    # Check if the dataset exists
    if not dataset_path.is_dir() or not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        exit(1)

    n_pillars_x = int(((args.x_max - args.x_min) / args.grid_cell_size))
    n_pillars_y = int(((args.y_max - args.y_min) / args.grid_cell_size))

    model = FastFlow3DModel(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y, point_features=8,
                            learning_rate=args.learning_rate)
    # waymo_data_module = WaymoDataModule(dataset_path, grid_cell_size=args.grid_cell_size, x_min=args.x_min,
    #                                     x_max=args.x_max, y_min=args.y_min,
    #                                     y_max=args.y_max, z_min=args.z_min, z_max=args.z_max,
    #                                     batch_size=args.batch_size,
    #                                     has_test=args.test_data_available,
    #                                     num_workers=args.num_workers)
    waymo_data_module = RandomDataModule(dataset_path, grid_cell_size=args.grid_cell_size, x_min=args.x_min,
                                         x_max=args.x_max, y_min=args.y_min,
                                         y_max=args.y_max, z_min=args.z_min, z_max=args.z_max,
                                         batch_size=args.batch_size,
                                         has_test=args.test_data_available,
                                         num_workers=args.num_workers)

    # Max epochs can be configured here to, early stopping is also configurable.
    # Some things are definable as callback from pytorch_lightning.callback
    trainer = pl.Trainer.from_argparse_args(args,
                                            progress_bar_refresh_rate=25,  # Prevents Google Colab crashes
                                            gpus=1 if torch.cuda.is_available() else 0
                                            )  # Add Trainer hparams if desired
    # The actual train loop
    trainer.fit(model, waymo_data_module)

    # Run also the testing
    if args.test_data_available and not args.fast_dev_run:
        trainer.test()  # Also loads the best checkpoint automatically


if __name__ == '__main__':
    cli()
