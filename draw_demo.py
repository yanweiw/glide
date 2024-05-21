#! /usr/bin/env python3
# Author : Yanwei Wang
# Email  : felixw@mit.edu
#
# Distributed under terms of the MIT license.

import matplotlib.pyplot as plt
import numpy as np
from polygon_mode_learning.two_polygons import TwoPolygons
from polygon_mode_learning.data_utils import seq2df, resample_traj
import pandas as pd
import random


class TrajDrawer:
    def __init__(self, task):
        self.task = task
        self.fig, self.ax = task.build_map()
        plt.show(block=False)
        self.ax.set_title('Click & Drag to draw, C to clear, Q when done')

        self.trajectories = []
        self.current_trajectory = []
        self.color = np.random.rand(3,)
        self.line, = self.ax.plot([], [], marker='o', linestyle='-', lw=0, markersize=2)
        self.lines = []
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_motion(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            self.current_trajectory.append([x, y])
            xdata = [x[0] for x in self.current_trajectory]
            ydata = [x[1] for x in self.current_trajectory]
            self.line.set_data(xdata, ydata)
            self.line.set_color(self.color)
            self.ax.draw_artist(self.line)
            self.fig.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        if event.button == 1:
            try:
                self.trajectories.append(np.vstack(self.current_trajectory).T)
                self.current_trajectory = []
                self.lines.append(self.line)
                self.line, = self.ax.plot([], [], marker='o', linestyle='-', lw=0, markersize=2)
                self.color = np.random.rand(3,)
            except:  # single point recorded, cannot vstack
                pass

    def on_key(self, event):
        if event.key == 'c':
            self.trajectories = []
            self.current_trajectory = []
            while len(self.lines) > 0:
                line = self.lines.pop()
                line.remove()
            self.fig.canvas.draw_idle()

    def get_trajectories(self):
        return self.trajectories


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pts", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_poly", type=int, default=2)
    args = parser.parse_args()

    # Draw trajs
    random.seed(args.seed)
    np.random.seed(args.seed)
    task = TwoPolygons(args.n_pts, num_polygons=args.n_poly)
    td = TrajDrawer(task)
    plt.show(block=True)
    trajs = td.get_trajectories()

    # Resample trajectories to fixed-length array
    for i, traj in enumerate(trajs):
        trajs[i] = resample_traj(traj, task._cfg["n_pts"])

    # Drop data if out-of-bound
    nan_idcs = []
    for i, traj in enumerate(trajs):
        if np.any(np.isnan(traj)):
            nan_idcs.append(i)

    if len(nan_idcs) > 0:
        print(f"Drop {len(nan_idcs)} out-of-bound trajectories.")
        trajs = [v for i, v in enumerate(trajs) if i not in nan_idcs]

    # Check traversal in task graph and store to data collection
    for traj in trajs:
        success, mode_seq, valid_transitions = task.validify_traj(traj)

        item = dict(traj_id=task._traj_id, x=traj[0].astype('float64'), y=traj[1].astype('float64'), success=success)
        item = {**item, **valid_transitions}
        item = seq2df(item)

        task.data = pd.concat([task.data, item], ignore_index=True)
        task._traj_id += 1

    # Plot drawn data
    task.show()

    # Save data
    if args.out_dir is not None:
        task.dump_data(args.out_dir, store_traj_vis=True)
        print(f'Data saved to {args.out_dir}')

