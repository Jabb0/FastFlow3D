from argparse import ArgumentParser

import open3d.visualization

from models import FastFlow3DModel
from data.WaymoDataset import WaymoDataset
import open3d as o3d
import ffmpeg
import time
import numpy as np
from utils.plot import visualize_point_cloud

# Open3D info
# http://open3d.org/html/tutorial/Basic/visualization.html
# http://www.open3d.org/docs/0.9.0/tutorial/Advanced/customized_visualization.html

if __name__ == '__main__':
    parser = ArgumentParser()

    # NOTE: IF MODEL IS NONE IT WILL VISUALIZE GROUND TRUTH DATA
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--data_directory', type=str)

    # start_frame and end_frame allow us just visualize a set of frames
    parser.add_argument('--start_frame', default=0, type=int)
    parser.add_argument('--end_frame', default=None, type=int)

    # Other paramaters
    screenshots_folder = "screenshots/temp_%04d.jpg"
    video_name = "point_cloud.mp4"
    debug = False

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
        model = FastFlow3DModel.load_from_checkpoint(args.model_path)
        model.eval()
        print("DISPLAYING PREDICTED DATA")
    else:
        print("DISPLAYING GROUND TRUTH DATA - NO MODEL HAS BEEN LOADED")


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    point_cloud = o3d.geometry.PointCloud()
    for i in range(args.start_frame, args.end_frame):
        print(f"Rendering frame {i} of {args.end_frame}")

        (previous_frame, current_frame), flows = waymo_dataset[i]

        if args.model_path is not None:
            flows = model((previous_frame, current_frame)).data.cpu().numpy()

        vis.add_geometry(point_cloud)

        ctr = vis.get_view_control()
        #ctr.set_zoom(0.72)
        #ctr.change_field_of_view(60.0)
        #ctr.set_front([ 0.42149238953069712, -0.81138370875554688, 0.40496992818454702 ])
        #ctr.set_lookat([ 13.056647215758161, 2.4109030723596945, 2.263514128894637 ])
        #ctr.set_up([ -0.25052622313468664, 0.32500897579092991, 0.91192421679501412 ])

        ctr.set_zoom(0.16)
        ctr.change_field_of_view(60.0)
        ctr.set_front([0.91296513629581766, -0.12678407628885097, 0.38784076359756481])
        ctr.set_lookat([-0.37671086910276924, 16.208793815613213, -3.1170728720176699])
        ctr.set_up([-0.39263835854591367, -0.01430683101675858, 0.91958166248823692])

        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(screenshots_folder % i, True)

        raw_point_cloud = current_frame[:, 0:3]
        point_cloud.points = o3d.utility.Vector3dVector(raw_point_cloud)
        if debug:
            visualize_point_cloud(raw_point_cloud)
        rgb_flow = flows[:, :-1]
        magnitudes = np.sqrt((rgb_flow ** 2).sum(-1))[..., np.newaxis]
        rgb_flow /= np.sqrt((rgb_flow ** 2).sum(-1))[..., np.newaxis]
        point_cloud.colors = o3d.utility.Vector3dVector(rgb_flow)

    (
        ffmpeg
            .input(screenshots_folder, framerate=25)
            .output(video_name)
            .run()
    )





