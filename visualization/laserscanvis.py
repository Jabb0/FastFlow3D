#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# https://github.com/PRBonn/lidar-bonnetal/blob/5a5f4b180117b08879ec97a3a05a3838bce6bb0f/train/common/laserscanvis.py

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import torch
from data.util import custom_collate_batch


class LaserScanVis:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self, dataset, start_frame=0, end_frame=None, model=None):
        # If model is None then it displays ground truth
        self.dataset = dataset  # This must be a configured Waymo dataset
        self.offset = start_frame
        self.end_frame = end_frame
        self.model = model

        self.mins, self.maxs = self.dataset.get_flow_ranges()

        self.reset()
        self.update_scan()

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)
        self.scan_vis = visuals.Markers()
        self.scan_view.camera = 'turntable'
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)


        # img canvas size
        self.multiplier = 1
        self.canvas_W = 1024
        self.canvas_H = 64

        # new canvas for img
        self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                      size=(self.canvas_W, self.canvas_H * self.multiplier))
        # grid
        self.img_grid = self.img_canvas.central_widget.add_grid()
        # interface (n next, b back, q quit, very simple)
        self.img_canvas.events.key_press.connect(self.key_press)
        self.img_canvas.events.draw.connect(self.draw)

        # add a view for the depth
        self.img_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.img_canvas.scene)
        self.img_grid.add_widget(self.img_view, 0, 0)
        self.img_vis = visuals.Image(cmap='viridis')
        self.img_view.add(self.img_vis)


    def update_scan(self):
        # first open data
        self.dataset.pillarize(False)
        (previous_frame, current_frame), flows = self.dataset[self.offset]
        flows = flows[:, :-1]  # Remove the label
        raw_point_cloud = current_frame[:, 0:3]
        # raw_point_cloud = current_frame[0][:, 0:3]
        if self.model is not None:
            self.dataset.pillarize(True)
            (previous_frame, current_frame), flows = self.dataset[self.offset]
            # We set batchsize of 1 for predictions
            batch = custom_collate_batch([((previous_frame, current_frame), flows)])
            with torch.no_grad():
                #output = self.model((previous_frame_tensor, current_frame_tensor))
                output = self.model(batch[0])
            flows = output[0].data.cpu().numpy()

        # then change names
        title = "scan " + str(self.offset) + " of Waymo"
        self.canvas.title = title
        self.img_canvas.title = title

        rgb_flow = (flows - self.mins) / (self.maxs - self.mins)
        # Need of clamping to 0 and 1, since may flow predictions can exceed it
        rgb_flow = np.clip(rgb_flow, a_min=0., a_max=1.)

        self.scan_vis.set_data(raw_point_cloud,
                               face_color=rgb_flow,
                               edge_color=rgb_flow,
                               size=1)


        self.img_vis.update()

    # interface
    def key_press(self, event):
        self.canvas.events.key_press.block()
        self.img_canvas.events.key_press.block()
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
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()
        if self.img_canvas.events.key_press.blocked():
            self.img_canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        self.img_canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()
