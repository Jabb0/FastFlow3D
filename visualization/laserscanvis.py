#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# https://github.com/PRBonn/lidar-bonnetal/blob/5a5f4b180117b08879ec97a3a05a3838bce6bb0f/train/common/laserscanvis.py

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import torch
from data.util import custom_collate_batch
import cv2


class LaserScanVis:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self, dataset, start_frame=0, end_frame=None, model=None, vis_previous_current=True):
        # If model is None then it displays ground truth
        self.dataset = dataset  # This must be a configured Waymo dataset
        self.offset = start_frame
        self.end_frame = end_frame
        self.model = model
        self.vis_previous_current = vis_previous_current  # True if you want to display 2 point clouds instead flows
        if model is not None:
            self.model.eval()

        self.mins, self.maxs = self.dataset.get_flow_ranges()

        self.reset()
        self.update_scan()

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities


        # --- Canvas for ground truth ---
        self.gt_canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.gt_canvas.events.key_press.connect(self.key_press)
        self.gt_canvas.events.draw.connect(self.draw)
        # grid
        self.gt_grid = self.gt_canvas.central_widget.add_grid()

        # laserscan part
        self.gt_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.gt_canvas.scene)
        self.gt_grid.add_widget(self.gt_view, 0, 0)
        self.gt_vis = visuals.Markers()
        self.gt_view.camera = 'turntable'
        self.gt_view.add(self.gt_vis)
        visuals.XYZAxis(parent=self.gt_view.scene)

        # --- Canvas por prediction ---
        self.predicted_canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.predicted_canvas.events.key_press.connect(self.key_press)
        self.predicted_canvas.events.draw.connect(self.draw)
        # grid
        self.predicted_grid = self.predicted_canvas.central_widget.add_grid()

        self.predicted_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.predicted_canvas.scene)
        self.predicted_grid.add_widget(self.predicted_view, 0, 0)
        self.predicted_vis = visuals.Markers()
        self.predicted_view.camera = 'turntable'
        self.predicted_view.add(self.predicted_vis)
        visuals.XYZAxis(parent=self.predicted_view.scene)

    def flow_to_rgb(self, flows):
        """
        Convert a flow to a rgb value
        Args:
            flows: (N, 3) vector flow

        Returns: (N, 3) RGB values normalized between 0 and 1

        """
        # https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
        # Use Hue, Saturation, Value colour model
        hsv = np.zeros((flows.shape[0], 1, 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flows[..., 0], flows[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        rgb = rgb[:, 0, :] / 255.  # Normalize to 1
        rgb[rgb < 0.2] = 0.2  # Just for visualize not moving points

        return rgb

    def update_scan(self):
        # first open data
        self.dataset.pillarize(False)
        (previous_frame, current_frame), flows = self.dataset[self.offset]
        gt_flows = flows[:, :-1]  # Remove the label
        raw_point_cloud = current_frame[:, 0:3]
        raw_point_cloud_previous = previous_frame[:, 0:3]
        # raw_point_cloud = current_frame[0][:, 0:3]
        if self.model is not None:  # Display predicted values
            self.dataset.pillarize(True)
            (previous_frame, current_frame), flows = self.dataset[self.offset]
            # We set batchsize of 1 for predictions
            batch = custom_collate_batch([((previous_frame, current_frame), flows)])
            with torch.no_grad():
                output = self.model(batch[0])
            predicted_flows = output[0].data.cpu().numpy()

        # then change names
        self.predicted_canvas.title = "Predicted frame " + str(self.offset) + " of Waymo"
        self.gt_canvas.title = "Ground truth frame " + str(self.offset) + " of Waymo"

        rgb_flow = self.flow_to_rgb(gt_flows)

        if self.vis_previous_current:
            concatenated_point_cloud = np.concatenate((raw_point_cloud_previous, raw_point_cloud))
            red = np.ones(raw_point_cloud_previous.shape) * np.array([1, 0, 0])
            green = np.ones(raw_point_cloud.shape) * np.array([0, 1, 0])
            concatenated_colors = np.concatenate((red, green))
            self.predicted_vis.set_data(concatenated_point_cloud,
                                   face_color=concatenated_colors,
                                   edge_color=concatenated_colors,
                                   size=1)

        else:
            if self.model is not None:
                rgb_flow_predicted = self.flow_to_rgb(predicted_flows)
                self.predicted_vis.set_data(raw_point_cloud,
                                       face_color=rgb_flow_predicted,
                                       edge_color=rgb_flow_predicted,
                                       size=1)
                self.predicted_vis.update()

            self.gt_vis.set_data(raw_point_cloud,
                                   face_color=rgb_flow,
                                   edge_color=rgb_flow,
                                   size=1)

        self.gt_vis.update()

    # interface
    def key_press(self, event):
        self.predicted_canvas.events.key_press.block()
        self.gt_canvas.events.key_press.block()
        if event.key == 'N':
            if self.offset < (self.end_frame - 1) and self.offset < len(self.dataset) - 1:
                self.offset += 1
            else:
                print("Maximum frame reached")
            self.update_scan()
        elif event.key == 'B':
            self.offset -= 1
            self.update_scan()
        elif event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def draw(self, event):
        if self.predicted_canvas.events.key_press.blocked():
            self.predicted_canvas.events.key_press.unblock()
        if self.gt_canvas.events.key_press.blocked():
            self.gt_canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.predicted_canvas.close()
        self.gt_canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()
