from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import os
import wandb
from pytorch_lightning.loggers import WandbLogger

from data import WaymoDataModule
from models import FastFlow3DModel, FastFlow3DModelScatter
from models.Flow3DModel import Flow3DModelV2
from utils import str2bool


def get_args():
    """
    Setup all arguments and parse them from commandline.
    :return: The ArgParser args object with everything parsed.
    """
    parser = ArgumentParser()
    parser.add_argument('data_directory', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--full_batch_size', default=None, type=int)
    parser.add_argument('--x_max', default=85, type=float)
    parser.add_argument('--x_min', default=-85, type=float)
    parser.add_argument('--y_max', default=85, type=float)
    parser.add_argument('--y_min', default=-85, type=float)
    parser.add_argument('--z_max', default=3, type=float)
    parser.add_argument('--z_min', default=-3, type=float)
    parser.add_argument('--grid_size', default=512, type=int)
    parser.add_argument('--test_data_available', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--fast_dev_run', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--wandb_enable', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_project', default="fastflow3d", type=str)
    parser.add_argument('--wandb_entity', default='dllab21fastflow3d', type=str)
    parser.add_argument('--use_sparse_lookup', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--architecture', default='FastFlowNet', type=str)
    parser.add_argument('--resume_from_checkpoint', type=str)
    parser.add_argument('--max_time', type=str)

    temp_args, _ = parser.parse_known_args()
    # Add the correct model specific args
    if temp_args.architecture == 'FastFlowNet':
        if temp_args.use_sparse_lookup:
            parser = FastFlow3DModel.add_model_specific_args(parser)
        else:
            parser = FastFlow3DModelScatter.add_model_specific_args(parser)
    elif temp_args.architecture == 'FlowNet':  # baseline
        parser = Flow3DModelV2.add_model_specific_args(parser)
    else:
        raise ValueError("no architecture {0} implemented".format(temp_args.architecture))

    # NOTE: Readd this to see all parameters of the trainer
    # parser = pl.Trainer.add_argparse_args(parser)  # Add arguments for the trainer
    # Add model specific arguments here
    return parser.parse_args()


def cli():
    args = get_args()

    if args.use_sparse_lookup and not args.fast_dev_run:
        print(f"ERROR: Sparse model is not implemented completely. No full run allowed")
        exit(1)

    if args.use_group_norm:
        print("INFO: Using group norm instead of batch norm!")

    dataset_path = Path(args.data_directory)
    # Check if the dataset exists
    if not dataset_path.is_dir() or not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        exit(1)

    # We assume, that the length of the grid is the same in x and y direction.
    # Otherwise, we have to implement different grid_cell_sizes for x and y direction
    if args.x_max + abs(args.x_min) != args.y_max + abs(args.y_min):
        raise ValueError("Grid must have same length in x and y direction but has a length of {0} in "
                         "x direction and {1} in y direction".format(args.x_max + abs(args.x_min),
                                                                     args.y_max + abs(args.y_min)))

    grid_cell_size = (args.x_max + abs(args.x_min)) / args.grid_size

    n_pillars_x = args.grid_size
    n_pillars_y = args.grid_size

    if args.architecture == 'FastFlowNet':
        if args.use_sparse_lookup:
            # Tested GPU memory increase from batch size 1 to 2 is 2350MiB
            model = FastFlow3DModel(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y, point_features=8,
                                    learning_rate=args.learning_rate)
        else:
            # Tested GPU memory increase from batch size 1 to 2 is 1824MiB
            model = FastFlow3DModelScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y,
                                           background_weight=args.background_weight, point_features=8,
                                           learning_rate=args.learning_rate,
                                           use_group_norm=args.use_group_norm)

    elif args.architecture == 'FlowNet':  # baseline
        model = Flow3DModelV2(learning_rate=args.learning_rate)
    else:
        raise ValueError("no architecture {0} implemented".format(args.architecture))
    waymo_data_module = WaymoDataModule(dataset_path, grid_cell_size=grid_cell_size, x_min=args.x_min,
                                        x_max=args.x_max, y_min=args.y_min,
                                        y_max=args.y_max, z_min=args.z_min, z_max=args.z_max,
                                        batch_size=args.batch_size,
                                        has_test=args.test_data_available,
                                        num_workers=args.num_workers,
                                        scatter_collate=not args.use_sparse_lookup,
                                        n_pillars_x=n_pillars_x)

    # Initialize the weights and biases logger.
    # Name is the name of this run
    # Project is the name of the project
    # Entity is the name of the team
    logger = True  # Not set a logger defaulting to tensorboard
    if args.wandb_enable:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            print("No WandB API key found in env: Set WANDB_API_KEY")
            exit(1)

        wandb.login(key=wandb_api_key)
        logger = WandbLogger(name=args.experiment_name, project=args.wandb_project, entity=args.wandb_entity)
        additional_hyperparameters = {'grid_cell_size': grid_cell_size,
                                      'x_min': args.x_min,
                                      'x_max': args.x_max,
                                      'y_max': args.y_max,
                                      'y_min': args.y_min,
                                      'z_min': args.z_min,
                                      'z_max': args.z_max,
                                      'n_pillars_x': n_pillars_x,
                                      'n_pillars_y': n_pillars_y,
                                      'batch_size': args.batch_size,
                                      'full_batch_size': args.full_batch_size,
                                      'has_test': args.test_data_available,
                                      'num_workers': args.num_workers,
                                      'scatter_collate': args.use_sparse_lookup
                                      }
        logger.log_hyperparams(additional_hyperparameters)
    else:
        print("No weights and biases API key set. Using tensorboard instead!")

    gradient_batch_acc = 1  # Do not accumulate batches before performing optimizer step
    if args.full_batch_size is not None:
        gradient_batch_acc = int(args.full_batch_size / args.batch_size)
        print(f"A full batch size is specified. The model will perform gradient update after {gradient_batch_acc} "
              f"smaller batches of size {args.batch_size} to approx. total batch size of {args.full_batch_size}."
              f"PLEASE NOTE that if the network includes layers that need larger batch sizes such as BatchNorm "
              f"they are still computed for each forward pass.")

    # Max epochs can be configured here to, early stopping is also configurable.
    # Some things are definable as callback from pytorch_lightning.callback
    trainer = pl.Trainer.from_argparse_args(args,
                                            precision=32,  # Precision 16 does not seem to work with batchNorm1D
                                            progress_bar_refresh_rate=25,  # Prevents Google Colab crashes
                                            gpus=1 if torch.cuda.is_available() else 0,  # -1 means "all GPUs"
                                            logger=logger,
                                            accumulate_grad_batches=gradient_batch_acc,
                                            log_every_n_steps=5
                                            )  # Add Trainer hparams if desired
    # The actual train loop
    trainer.fit(model, waymo_data_module)

    # Run also the testing
    if args.test_data_available and not args.fast_dev_run:
        trainer.test()  # Also loads the best checkpoint automatically


if __name__ == '__main__':
    cli()
