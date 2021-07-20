import time
from pathlib import Path
from argparse import ArgumentParser

from data.preprocess import preprocess, merge_metadata
from data.util import generate_flying_things_point_cloud, get_all_flying_things_frames
import os
import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

global output_directory


def preprocess_wrap(tfrecord_file):
    preprocess(tfrecord_file, output_directory, frames_per_segment=None)


def preprocess_flying_things(input_dir, output_dir, view='right'):
    """
    Data directory must be in shape of:

    parent-dir
        disparity
            train
               left
               right
            val
                left
               right
        disparity_change
            train
               left
               right
            val
                left
               right
        optical_flow
            train
                backward
                    left
                    right
                forward
                   left
                   right
            val
                backward
                    left
                    right
                forward
                   left
                   right
    """
    # INPUT_DIR = "./data/flyingthings3d"
    # OUTPUT_DIR = "./data/flyingthings3d_preprocessed"

    all_files_disparity, all_files_disparity_change, all_files_opt_flow = get_all_flying_things_frames(
        input_dir=input_dir, disp_dir='disparity/train/{}'.format(view),
        opt_dir='optical_flow/train/backward/{}'.format(view), disp_change_dir='disparity_change/train/{}'.format(view))

    for i in range(len(all_files_disparity) - 1):
        disparity = all_files_disparity[i]
        disparity_next_frame = all_files_disparity[i + 1]
        disparity_change = all_files_disparity_change[i]
        optical_flow = all_files_opt_flow[i]
        d = generate_flying_things_point_cloud(disparity, disparity_next_frame, disparity_change, optical_flow)
        np.savez_compressed(
            os.path.join(output_dir, 'frame_{}.npz'.format(i)), points1=d[0], points2=d[1], flow=d[2], mask=d[3])


# https://github.com/tqdm/tqdm/issues/484
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_directory', type=str)
    parser.add_argument('output_directory', type=str)
    parser.add_argument('--n_cores', default=None, type=int)
    parser.add_argument('--dataset', default='waymo', type=str)
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
    output_directory = os.path.abspath(output_directory)

    # TODO also use multiple cores for preprocessing flying things dataset (?)
    if args.dataset == 'waymo':
        n_cores = mp.cpu_count()
        if args.n_cores is not None:
            if args.n_cores <= 0:
                print("Number of cores cannot be negative")
                exit(1)
            if args.n_cores > n_cores:
                print(f"Number of cores cannot be more than{n_cores}")
                exit(1)
            else:
                n_cores = args.n_cores

        print(f"{n_cores} number of cores available")

        pool = mp.Pool(n_cores)

        tfrecord_filenames = []
        os.chdir(input_directory)
        for file in glob.glob("*.tfrecord"):
            file_name = os.path.abspath(file)
            tfrecord_filenames.append(file_name)

        t = time.time()

        for _ in tqdm(pool.imap_unordered(preprocess_wrap, tfrecord_filenames), total=len(tfrecord_filenames)):
            pass

        # Close Pool and let all the processes complete
        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

        # Merge look up tables
        print("Merging individual metadata...")
        merge_metadata(os.path.abspath(output_directory))

        print(f"Preprocessing duration: {(time.time() - t):.2f} s")
    elif args.dataset == 'flying_things':
        preprocess_flying_things(input_dir=input_directory, output_dir=output_directory)
    else:
        raise ValueError('Dataset {} not available'.format(args.dataset))


