import tensorflow as tf
import time
from pathlib import Path
from argparse import ArgumentParser

from data.WaymoDataset import WaymoDataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_directory', type=str)
    parser.add_argument('output_directory', type=str)
    args = parser.parse_args()

    print(f"Extracting frames from {args.input_directory} to {args.output_directory}")

    input_directory = Path(args.input_directory)
    if not input_directory.exists() or not input_directory.is_dir():
        print("Input directory does not exist")
        exit(1)
    output_directory = Path(args.output_directory)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    if list(output_directory.iterdir()):
        print("Output directory not empty! Please remove existing files as there is no merge.")
        exit(1)

    # Getting points with the dataloader
    preprocess = True
    t = time.time()
    waymo_dataset = WaymoDataset(args.output_directory, force_preprocess=preprocess, tfrecord_path=args.input_directory,
                                 drop_invalid_point_function=None, point_cloud_transform=None,
                                 frames_per_segment=100)  # Take 100 Frames from each segment
    print(f"Preprocessing duration: {(time.time() - t):.2f} s")

    exit(0)
    accum = 0
    t = time.time()
    for item in waymo_dataset:
        accum += (time.time() - t)
        t = time.time()
    print(f"Mean access time: {(accum/len(waymo_dataset)):.2f} s")


