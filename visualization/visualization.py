from argparse import ArgumentParser

from data.WaymoDataset import WaymoDataset

import yaml
from data.util import ApplyPillarization, drop_points_function
from laserscanvis import LaserScanVis
from models.FastFlow3DModelScatter import FastFlow3DModelScatter


# vispy
# if error vispy:
# https://askubuntu.com/questions/308128/failed-to-load-platform-plugin-xcb-while-launching-qt5-app-on-linux-without
# https://gist.github.com/ujjwal96/1dcd57542bdaf3c9d1b0dd526ccd44ff


if __name__ == '__main__':
    parser = ArgumentParser()

    # NOTE: IF MODEL IS NONE IT WILL VISUALIZE GROUND TRUTH DATA
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--config_file', default=None, type=str)
    parser.add_argument('--data_directory', type=str)
    # TODO this not work if you pass the parameter False
    parser.add_argument('--pillarization', default=False, type=bool)

    # start_frame and end_frame allow us just visualize a set of frames
    parser.add_argument('--start_frame', default=0, type=int)
    parser.add_argument('--end_frame', default=None, type=int)

    args = parser.parse_args()
    waymo_dataset = WaymoDataset(args.data_directory)

    if args.end_frame is None:
        args.end_frame = len(waymo_dataset)

    if args.start_frame < 0 or args.start_frame > len(waymo_dataset):
        raise ValueError("Start frame must be greater than 0 and less thant the dataset length")
    if args.end_frame < 0 or args.end_frame > len(waymo_dataset):
        raise ValueError("End frame must be greater than 0 and less thant the dataset length")
    if args.start_frame > args.end_frame:
        raise ValueError("Start frame cannot be greater than end frame")

    if args.model_path is not None:
        # We assume 512x512 pillars grid and 8 features per point
        model = FastFlow3DModelScatter.load_from_checkpoint(args.model_path)
        model.eval()
        print("DISPLAYING PREDICTED DATA")
    else:
        model = None
        print("DISPLAYING GROUND TRUTH DATA - NO MODEL HAS BEEN LOADED")


    # Load config file (must be downloaded from Weights and Biases), it has the name of config.yaml
    with open(args.config_file, 'r') as stream:
        try:
            config_info = yaml.safe_load(stream)
            grid_cell_size = config_info['grid_cell_size']['value']
            x_min = config_info['x_min']['value']
            y_min = config_info['y_min']['value']
            z_min = config_info['z_min']['value']
            z_max = config_info['z_max']['value']
            x_max = config_info['x_max']['value']
            y_max = config_info['y_max']['value']
            # TODO save number of pillars in config file
            point_cloud_transform = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=x_min,
                                                       y_min=y_min, z_min=z_min, z_max=z_max, n_pillars_x=512)
            if args.pillarization:
                waymo_dataset.set_point_cloud_transform(point_cloud_transform)
            drop_points_function = drop_points_function(x_min=x_min,
                                                        x_max=x_max, y_min=y_min, y_max=y_max,
                                                        z_min=z_min, z_max=z_max)
            waymo_dataset.set_drop_invalid_point_function(drop_points_function)

        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    vis = LaserScanVis(dataset=waymo_dataset,
                       start_frame=args.start_frame,
                       end_frame=args.end_frame,
                       model=model,
                       pillarization=args.pillarization)
    vis.run()




