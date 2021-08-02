import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from data import WaymoDataModule
from data.FlyingThings3DDataModule import FlyingThings3DDataModule
from models import FastFlow3DModelScatter
from utils import str2bool


def get_args():
    """
    Setup all arguments and parse them from commandline.
    :return: The ArgParser args object with everything parsed.
    """
    parser = ArgumentParser(description="Training script for FastFlowNet and FlowNet3D "
                                        "based on Waymo or flying thing dataset",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # Required arguments
    parser.add_argument('data_directory', type=str, help="Path to the data directory. "
                                                         "Needs to have preprocessed directories "
                                                         "train and valid inside.")
    parser.add_argument('experiment_name', type=str, help="Name of the experiment for logging purposes.")
    # Model related arguments
    parser.add_argument('--architecture',
                        default='FastFlowNet',
                        choices=['FastFlowNet', 'FlowNet', 'FlowNetV2'],
                        help="The model architecture to use")
    parser.add_argument('--resume_from_checkpoint', type=str,
                        help="Path to ckpt file to resume from. Parameter from PytorchLightning Trainer.")
    # Data related arguments
    parser.add_argument('--dataset', default='waymo',
                        choices=["waymo", 'flying_things'],
                        help="Dataset Type to train on.")
    parser.add_argument('--x_max', default=85, type=float, help="x boundary in positive direction")
    parser.add_argument('--x_min', default=-85, type=float, help="x boundary in negative direction")
    parser.add_argument('--y_max', default=85, type=float, help="y boundary in positive direction")
    parser.add_argument('--y_min', default=-85, type=float, help="y boundary in negative direction")
    parser.add_argument('--z_max', default=3, type=float, help="z boundary in positive direction")
    parser.add_argument('--z_min', default=-3, type=float, help="z boundary in negative direction")
    parser.add_argument('--grid_size', default=512, type=int, help="")
    parser.add_argument('--n_points', default=None, type=int,
                        help="Number of Points to use from each point cloud. Forces downsampling.")
    parser.add_argument('--test_data_available', type=str2bool, nargs='?', const=True, default=False,
                        help="If dataset path also has test directory and to use it.")
    # Global training parameters
    parser.add_argument('--batch_size', default=2, type=int, help="Batch size each GPU trains on.")
    parser.add_argument('--full_batch_size', default=None, type=int,
                        help="Batch size for each GPU after which the gradient update should happen.")
    # Logging related parameters
    parser.add_argument('--wandb_enable', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_project', default="fastflow3d", type=str)
    parser.add_argument('--wandb_entity', default='dllab21fastflow3d', type=str)
    parser.add_argument('--wandb_run_id', default=None, type=str,
                        help="Id of an existing WnB run that should be resumed.")  # Id of the run
    # Dev parameters
    parser.add_argument('--only_valid', type=str2bool, nargs='?', const=True, default=False,
                        help="Only run trainer.validate instead of trainer.fit")
    parser.add_argument('--fast_dev_run', type=str2bool, nargs='?', const=True, default=False)
    # Training machine related parameters
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--max_time', type=str, help="Max time after which to stop training.")
    # Parameters for multi gpu training
    parser.add_argument('--gpus', default=1, type=int, help="GPU parameter of PL Trainer class.")
    parser.add_argument('--accelerator', default=None, type=str,
                        help="Accelerator to use. Set to ddp for multi GPU training.")  # Param of Trainer
    parser.add_argument('--sync_batchnorm', type=str2bool, default=False, nargs='?', const=True,
                        help="Whether to use sync batchnorm in multi GPU training. "
                             "Defaults to False but recommended to turn on if batch_size is too small.")
    parser.add_argument('--disable_ddp_unused_check', type=str2bool, default=False, nargs='?', const=True,
                        help="Disable unused parameter check for ddp to improve speed. "
                             "See https://pytorch-lightning.readthedocs.io/en/stable/benchmarking/"
                             "performance.html#when-using-ddp-set-find-unused-parameters-false")

    # NOTE: Readd this to see all parameters of the trainer
    # parser = pl.Trainer.add_argparse_args(parser)  # Add arguments for the trainer

    # TODO: This does not show the arguments in --help properly.
    temp_args, _ = parser.parse_known_args()
    # Add the correct model specific args
    if temp_args.architecture == 'FastFlowNet':
        parser = FastFlow3DModelScatter.add_model_specific_args(parser)
    elif temp_args.architecture == 'FlowNet':  # baseline
        from models.Flow3DModel import Flow3DModel
        parser = Flow3DModel.add_model_specific_args(parser)
    elif temp_args.architecture == 'FlowNetV2':  # baseline
        from models.Flow3DModel import Flow3DModelV2
        parser = Flow3DModelV2.add_model_specific_args(parser)
    else:
        raise ValueError("no architecture {0} implemented".format(temp_args.architecture))

    return parser.parse_args()


def cli():
    args = get_args()

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

    apply_pillarization = True

    interpolate_prediction = False if args.n_points is None else True

    if args.architecture == 'FastFlowNet':
        # Tested GPU memory increase from batch size 1 to 2 is 1824MiB
        model = FastFlow3DModelScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y,
                                       background_weight=args.background_weight, point_features=8,
                                       learning_rate=args.learning_rate,
                                       use_group_norm=args.use_group_norm,
                                       interpolate=interpolate_prediction)
    elif args.architecture == 'FlowNet':  # baseline
        apply_pillarization = False  # FlowNet does not use pillarization
        in_channels = 6 if args.dataset == 'flying_things' else 5  # TODO create cfg file?
        from models.Flow3DModel import Flow3DModel
        model = Flow3DModel(learning_rate=args.learning_rate, n_samples_set_up_conv=args.n_samples_set_up_conv,
                            n_samples_set_conv=args.n_samples_set_conv, n_samples_flow_emb=args.n_samples_flow_emb,
                            in_channels=in_channels, interpolate=interpolate_prediction)
    elif args.architecture == 'FlowNetV2':  # baseline
        apply_pillarization = False  # FlowNet does not use pillarization
        from models.Flow3DModel import Flow3DModelV2
        model = Flow3DModelV2(learning_rate=args.learning_rate)
    else:
        raise ValueError("no architecture {0} implemented".format(args.architecture))

    if args.dataset == 'waymo':
        data_module = WaymoDataModule(dataset_path, grid_cell_size=grid_cell_size, x_min=args.x_min,
                                      x_max=args.x_max, y_min=args.y_min,
                                      y_max=args.y_max, z_min=args.z_min, z_max=args.z_max,
                                      batch_size=args.batch_size,
                                      has_test=args.test_data_available,
                                      num_workers=args.num_workers,
                                      n_pillars_x=n_pillars_x,
                                      n_points=args.n_points, apply_pillarization=apply_pillarization)
    elif args.dataset == 'flying_things':
        data_module = FlyingThings3DDataModule(dataset_path,
                                               batch_size=args.batch_size,
                                               has_test=args.test_data_available,
                                               num_workers=args.num_workers,
                                               n_points=args.n_points)
    else:
        raise ValueError('Dataset {} not available'.format(args.dataset))

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

        if args.wandb_run_id is not None:
            print(f"Resuming with Weights and Biases run {args.wandb_run_id}")

        logger = WandbLogger(name=args.experiment_name, project=args.wandb_project, entity=args.wandb_entity,
                             log_model=True, id=args.wandb_run_id)
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
                                      'architecture': args.architecture,
                                      'n_points': args.n_points,
                                      'dataset': args.dataset
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

    plugins = None
    if args.disable_ddp_unused_check:
        if not args.accelerator == "ddp":
            print("FATAL: DDP unused checks can only be disabled when DDP is used as accelerator!")
            exit(1)
        print("Disabling unused parameter check for DDP")
        plugins = DDPPlugin(find_unused_parameters=False)

    # Add a callback for checkpointing after each epoch and the model with best validation loss
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)

    # Max epochs can be configured here to, early stopping is also configurable.
    # Some things are definable as callback from pytorch_lightning.callback
    trainer = pl.Trainer.from_argparse_args(args,
                                            precision=32,  # Precision 16 does not seem to work with batchNorm1D
                                            gpus=args.gpus if torch.cuda.is_available() else 0,  # -1 means "all GPUs"
                                            logger=logger,
                                            accumulate_grad_batches=gradient_batch_acc,
                                            log_every_n_steps=5,
                                            plugins=plugins,
                                            callbacks=[checkpoint_callback]
                                            )  # Add Trainer hparams if desired
    # The actual train loop
    if not args.only_valid:
        trainer.fit(model, data_module)
    else:
        trainer.validate(model=model, datamodule=data_module)

    # Run also the testing
    if args.test_data_available and not args.fast_dev_run:
        trainer.test()  # Also loads the best checkpoint automatically


if __name__ == '__main__':
    cli()
