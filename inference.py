from argparse import ArgumentParser
from models import FastFlow3DModel
from data.WaymoDataset import WaymoDataset
import open3d as o3d
import ffmpeg

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
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(screenshots_folder % i, True)

        raw_point_cloud = current_frame[:, 0:3]
        point_cloud.points = o3d.utility.Vector3dVector(raw_point_cloud)
        rgb_flow = flows[:, :-1]
        point_cloud.colors = o3d.utility.Vector3dVector(rgb_flow)

    (
        ffmpeg
            .input(screenshots_folder, framerate=25)
            .output(video_name)
            .run()
    )





