from argparse import ArgumentParser

import ffmpeg
import open3d as o3d

from data.WaymoDataset import WaymoDataset

# https://github.com/intel-isl/Open3D/issues/1110
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_directory', type=str)
    args = parser.parse_args()
    screenshots_folder = "screenshots/temp_%04d.jpg"
    video_name = "point_cloud.mp4"

    draw_flows = True

    print(f"Reading frames from {args.data_directory}")

    # Ensure that the working directory is the root project!
    waymo_dataset = WaymoDataset(args.data_directory)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    point_cloud = o3d.geometry.PointCloud()
    for i, item in enumerate(waymo_dataset):
        #if i < 99:
        #    continue
        #if i > 10:
        #    break

        vis.add_geometry(point_cloud)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(screenshots_folder % i, True)

        current = item[0][1]
        flows = item[1]
        raw_point_cloud = current[:, 0:3]
        if draw_flows:
            point_cloud.points = o3d.utility.Vector3dVector(raw_point_cloud)
            rgb_flow = flows[:, :-1]
            #rgb_flow /= np.sum(rgb_flow)
            point_cloud.colors = o3d.utility.Vector3dVector(rgb_flow)
        else:
            point_cloud.points = o3d.utility.Vector3dVector(raw_point_cloud)
        #vis.destroy_window()

    (
        ffmpeg
            .input(screenshots_folder, framerate=25)
            .output(video_name)
            .run()
    )