import torch
import numpy as np

from argparse import ArgumentParser
from pathlib import Path

from data import WaymoDataModule


def get_data(loader):
    total_samples = 0
    # We have six possible labels from -1 to 4
    total_counts = torch.zeros((6,))

    for i, batch in enumerate(loader):
        _, y = batch
        total_samples += y.size(0)
        labels = y[:, :, 3]
        idx, counts = labels.unique(return_counts=True)
        # Idx is from -1 to 4
        total_counts[idx.long() + 1] += counts.int()

        if i % 100 == 0:
            print(f"Batch {i}")

    np.set_printoptions(precision=3, suppress=True)

    print(f"Total samples {total_samples}")
    print(f"Total counts {total_counts.numpy()}")
    print(f"Total points {total_counts.sum()}")
    print(f"Percentages {((total_counts / total_counts.sum()) * 100).numpy()}")


def main():
    parser = ArgumentParser(description="Training script for FastFlowNet and FlowNet3D "
                                        "based on Waymo or flying thing dataset")
    # Required arguments
    parser.add_argument('data_directory', type=str, help="Path to the data directory. "
                                                         "Needs to have preprocessed directories "
                                                         "train and valid inside.")
    parser.add_argument('--x_max', default=85, type=float, help="x boundary in positive direction")
    parser.add_argument('--x_min', default=-85, type=float, help="x boundary in negative direction")
    parser.add_argument('--y_max', default=85, type=float, help="y boundary in positive direction")
    parser.add_argument('--y_min', default=-85, type=float, help="y boundary in negative direction")
    parser.add_argument('--z_max', default=3, type=float, help="z boundary in positive direction")
    parser.add_argument('--z_min', default=-3, type=float, help="z boundary in negative direction")
    parser.add_argument('--grid_size', default=512, type=int, help="")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size each GPU trains on.")
    parser.add_argument('--n_points', default=None, type=int,
                        help="Number of Points to use from each point cloud. Forces downsampling.")
    parser.add_argument('--num_workers', default=4, type=int)

    args = parser.parse_args()

    grid_cell_size = (args.x_max + abs(args.x_min)) / args.grid_size

    n_pillars_x = args.grid_size

    dataset_path = Path(args.data_directory)

    # Iterate the dataset and count the number of samples and the number of points per label
    data_module = WaymoDataModule(dataset_path, grid_cell_size=grid_cell_size, x_min=args.x_min,
                                  x_max=args.x_max, y_min=args.y_min,
                                  y_max=args.y_max, z_min=args.z_min, z_max=args.z_max,
                                  batch_size=args.batch_size,
                                  has_test=False,
                                  num_workers=args.num_workers,
                                  n_pillars_x=n_pillars_x,
                                  n_points=args.n_points, apply_pillarization=True,
                                  shuffle_train=False)  # Do not shuffle train for this
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    print("Train")
    get_data(train_dataloader)

    print()
    print("Val")
    val_dataloader = data_module.val_dataloader()
    get_data(val_dataloader)


if __name__ == '__main__':
    main()
