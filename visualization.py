from argparse import ArgumentParser

import yaml

from data.WaymoDataset import WaymoDataset
from data.util import ApplyPillarization, drop_points_function
from utils import str2bool
from visualization.util import predict_and_store_flows, flows_exist
from models.FastFlow3DModelScatter import FastFlow3DModelScatter

# vispy
# if error vispy:
# https://askubuntu.com/questions/308128/failed-to-load-platform-plugin-xcb-while-launching-qt5-app-on-linux-without
# https://gist.github.com/ujjwal96/1dcd57542bdaf3c9d1b0dd526ccd44ff
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('data_directory', type=str)
    parser.add_argument('config_file', type=str)
    # NOTE: IF MODEL IS NONE IT WILL VISUALIZE GROUND TRUTH DATA
    parser.add_argument('--model_path', default=None, type=str)

    # start_frame and end_frame allow us just visualize a set of frames
    parser.add_argument('--start_frame', default=0, type=int)
    parser.add_argument('--end_frame', default=None, type=int)
    parser.add_argument('--vis_previous_current', default=False, type=bool)

    # If you want online prediction or first predict, store the flows and predict them
    # This is suitable for slow systems since it reads the flows then from disk
    parser.add_argument('--online', type=str2bool, nargs='?', const=False, default=True)

    #  If you want to create an automatic video of the visualization
    # --video {gt, model}, gt if you want a video of the ground truth or model if you want a video of the model
    parser.add_argument('--video', default=None, type=str)

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

    # Load config file (must be downloaded from Weights and Biases), it has the name of config.yaml
    with open(args.config_file, 'r') as stream:
        try:
            config_info = yaml.safe_load(stream)
            grid_cell_size = config_info['grid_cell_size']['value']
            x_min = config_info['x_min']['value']
            y_min = config_info['y_min']['value']
            z_min = config_info['z_min']['value']
            x_max = config_info['x_max']['value']
            y_max = config_info['y_max']['value']
            z_max = config_info['z_max']['value']
            n_pillars_x = config_info['n_pillars_x']['value']
            n_pillars_y = config_info['n_pillars_y']['value']
            point_cloud_transform = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=x_min,
                                                       y_min=y_min, z_min=z_min, z_max=z_max, n_pillars_x=n_pillars_x)

            waymo_dataset.set_point_cloud_transform(point_cloud_transform)
            drop_points_function = drop_points_function(x_min=x_min,
                                                        x_max=x_max, y_min=y_min, y_max=y_max,
                                                        z_min=z_min, z_max=z_max)
            waymo_dataset.set_drop_invalid_point_function(drop_points_function)

            if "n_points" in config_info.keys():
                n_points = config_info['n_points']['value']
                if n_points is not None and n_points != 'None':
                    waymo_dataset.set_n_points(n_points)

            if "architecture" in config_info.keys():
                architecture = config_info['architecture']['value']
            else:
                architecture = "FastFlowNet"
            if args.model_path is not None and not flows_exist(waymo_dataset):
                if architecture == "FastFlowNet":
                    model = FastFlow3DModelScatter.load_from_checkpoint(args.model_path)
                    model.eval()
                    print("DISPLAYING PREDICTED DATA WITH FASTFLOWNET")
                elif architecture == "FlowNet":
                    from models.Flow3DModel import Flow3DModel
                    model = Flow3DModel.load_from_checkpoint(args.model_path)
                    model.cuda()
                    model.eval()
                    print("DISPLAYING PREDICTED DATA WITH FLOWNET (baseline)")
                else:
                    raise ValueError("no architecture {0} implemented".format(architecture))
            else:
                model = None
                print("DISPLAYING GROUND TRUTH DATA - NO MODEL HAS BEEN LOADED")

        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    if args.online is not True:
        # Predict and store into disk
        print(f"Predicting and storing {len(waymo_dataset)} frames...")
        predict_and_store_flows(model, waymo_dataset, architecture=architecture)

    from visualization.laserscanvis import LaserScanVis
    vis = LaserScanVis(dataset=waymo_dataset,
                       start_frame=args.start_frame,
                       end_frame=args.end_frame,
                       model=model,
                       vis_previous_current=args.vis_previous_current,
                       online=args.online,
                       video=args.video)
    vis.run()




