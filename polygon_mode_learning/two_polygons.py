#! /usr/bin/env python3
# Author : Tsun-Hsuan Wang
#
# Distributed under terms of the MIT license.

from typing import Tuple, Optional, Dict
import os
import shutil
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import pandas as pd
from polygenerator import random_convex_polygon
import pickle
import random

from polygon_mode_learning.data_types import Region, Mode, Transition
from polygon_mode_learning.data_utils import traj2modeseq, validify_transitions, arrow_handler_map, \
                    taskdef2graph, plot_graph

__all__ = ["TwoPolygons"]


class TwoPolygons:
    @staticmethod
    def _task_def(num_polygons=2):
        num_points = 4
        polygon_base = random_convex_polygon(num_points=num_points)
        polygons = [polygon_base]
        for _ in range(num_polygons-1):
            polygon_new = random_convex_polygon(num_points=num_points)
            polygon_new = attach_polygons(polygon_base, polygon_new)
            polygons.append(polygon_new)
            polygon_base = polygon_new

        for i, polygon in enumerate(polygons):
            polygon = np.array(polygon) * 5 - 4.
            polygons[i] = polygon

        modes = [Mode(id=0, name="free", is_initial=True)] # Mode 0 is free space, potentially non-convex in this case
        for i in range(len(polygons)):
            modes.append(Mode(id=i+1, name='mode_'+str(i+1), color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),\
                              region=Region(polygon=polygons[i]), is_initial=False, is_goal=((i+1)==len(polygons))))

        transitions = []
        for i in range(len(polygons)+1): # total num of modes = free mode + polygons
            transitions.append(Transition(inp=modes[i], out=modes[i])) # self-transitions

        for i in range(len(polygons)):
            transitions.append(Transition(inp=modes[i], out=modes[i+1])) # demonstration

        for i in range(len(polygons)):
            transitions.append(Transition(inp=modes[i+1], out=modes[i])) # reversability

        for i in range(len(polygons)-1): # duplicate of mode_1 -> mode_0
            transitions.append(Transition(inp=modes[i+2], out=modes[0])) # restartability

        required_paths = [
            [modes[len(polygons) - 1], modes[len(polygons)]]
        ]

        return modes, transitions, required_paths

    FIGSIZE = (6, 6)  # related to mode region
    XLIM = (-8, 8)
    YLIM = (-8, 8)

    def __init__(self, n_pts: int = 100, num_polygons: int = 2):
        modes, transitions, required_paths = TwoPolygons._task_def(num_polygons)

        self._cfg = {"n_pts": n_pts}
        self._modes = modes
        self._transitions = transitions
        self._required_paths = required_paths
        self.data = pd.DataFrame(columns=[
            "traj_id", # trajectory ID
            "t", # timestep along the trajectory
            "x", # x position of the waypoint
            "y", # y position of the waypoint
            "success", # trajectory-level success
        ] + \
        [f"transition_{v.inp.name}_to_{v.out.name}" for v in self._transitions] # transition-level success
        )
        self._traj_id = 0

    @property
    def modes(self):
        return self._modes

    @property
    def transitions(self):
        return self._transitions

    def build_map(self, fig_size=None, fig_ax=None) -> Tuple[plt.Figure, plt.Axes]:
        if fig_size is not None:
            self.FIGSIZE = fig_size
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=self.FIGSIZE)
        else:
            fig, ax = fig_ax
        ax.set_xlim(*self.XLIM)
        ax.set_ylim(*self.YLIM)
        for mode in self._modes:
            if mode.region is not None:
                patch_kwargs = dict(alpha=0.5, color=mode.color)
                patch = Polygon(mode.region.polygon, **patch_kwargs)
                ax.add_patch(patch)
        return fig, ax

    def validify_traj(self, traj: np.ndarray):
        mode_seq = traj2modeseq(traj, self._modes)
        success, valid_transitions = validify_transitions(mode_seq, self._transitions, self._required_paths)

        valid_transitions = np.concatenate([valid_transitions, valid_transitions[-1:]], axis=0)  # make size compatible (assume constate transition at the end)

        valid_transitions = {f"transition_{v.inp.name}_to_{v.out.name}": valid_transitions[:, i] for i, v in enumerate(self._transitions)}

        return success, mode_seq, valid_transitions

    def show(self):
        fig, ax = self.build_map()
        for traj_id, item in self.data.groupby("traj_id"):
            x = item["x"].to_numpy().astype('float64')
            y = item["y"].to_numpy().astype('float64')
            ax.plot(x, y)
        plt.show()
        plt.close(fig)

    def dump_data(self, out_dir: str, store_traj_vis: bool = False):
        os.makedirs(out_dir, exist_ok=True)

        # Save data
        fpath = os.path.join(out_dir, "data.csv")
        self.data.to_csv(fpath)

        # Save task graph
        G = taskdef2graph(self._modes, self._transitions)
        fpath = os.path.join(out_dir, "task_graph.png")
        plot_graph(G, fpath=fpath)

        # Save map
        fig, ax = self.build_map()
        fpath = os.path.join(out_dir, "map.png")
        fig.tight_layout()
        fig.savefig(fpath)

        # Save visualization of trajectories
        if store_traj_vis:
            traj_vis_dir = os.path.join(out_dir, "traj_vis")
            os.makedirs(traj_vis_dir, exist_ok=True)

            cmap = self.plot_legend_from_transitions(fig, ax)

            quiver = None
            for traj_id, item in self.data.groupby("traj_id"):
                quiver = self.plot_traj(item, fig, ax, cmap, quiver)

                fpath = os.path.join(traj_vis_dir, f"traj_{traj_id:05d}.png")
                fig.savefig(fpath)

        # Save code for future back-tractability
        code_out_dir = os.path.join(out_dir, "code")
        os.makedirs(code_out_dir, exist_ok=True)

        shutil.copy(__file__, os.path.join(code_out_dir, "task.py"))
        for fn in ["data_types.py", "data_utils.py"]:
            shutil.copy(os.path.join(os.path.dirname(__file__), fn), code_out_dir)

        with open(os.path.join(code_out_dir, "cfg.json"), "w") as f:
            json.dump(self._cfg, f, indent=4, sort_keys=True)

        # dump the task object
        with open(os.path.join(out_dir, "task.pkl"), 'wb') as outp:
            pickle.dump(self, outp)

    def plot_legend_from_transitions(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        return_legends_kwargs: Optional[bool] = False,
    ) -> mpl.colors.Colormap:
        cmap = mpl.cm.get_cmap("Set1")
        legends_kwargs = dict(
            handles=[],
            labels=[],
            prop=dict(size=6),
            handler_map=arrow_handler_map(),
            bbox_to_anchor=(0.5, 1.13),
            ncol=3,
            loc="upper center",
        )
        def add_legend(label, color):
            arrow = mpl.patches.FancyArrow(0, 0, 3, 3, color=color)
            legends_kwargs["handles"].append(arrow)
            legends_kwargs["labels"].append(label)
        add_legend("invalid", "k")
        for i, tr in enumerate(self._transitions):
            label = f"{tr.inp.name}->{tr.out.name}"
            add_legend(label, cmap(i))
        ax.legend(**legends_kwargs)
        fig.subplots_adjust(top=0.85)

        if return_legends_kwargs:
            return cmap, legends_kwargs
        else:
            return cmap

    def plot_traj(
        self,
        item: pd.Series,
        fig: plt.Figure,
        ax: plt.Axes,
        cmap: mpl.colors.Colormap,
        quiver: Optional[mpl.quiver.Quiver] = None,
    ) -> mpl.quiver.Quiver:
        if item.success.iloc[0]:
            fig.suptitle("Success")
        else:
            fig.suptitle("Fail")

        x = item["x"].to_numpy().astype('float64')
        y = item["y"].to_numpy().astype('float64')
        dx = (x[1:] - x[:-1]) * 0.5
        dy = (y[1:] - y[:-1]) * 0.5

        colors = []
        for i, v in item.iterrows():
            color = None
            for ii, tr in enumerate(self._transitions):
                is_tr = v[f"transition_{tr.inp.name}_to_{tr.out.name}"]
                if is_tr:
                    color = cmap(ii)
            if color is None:
                color = (0., 0., 0., 1.) # color for invalid transition
            colors.append(color)
        colors = colors[:-1] # drop the last copied transition

        if quiver is None:
            # import ipdb; ipdb.set_trace()
            quiver = ax.quiver(x[:-1], y[:-1], dx, dy, color=colors, scale=10)
        else:
            quiver.set_offsets(np.stack([x, y], axis=1)[:-1])
            quiver.set_UVC(dx, dy)
            quiver.set_edgecolors(colors)
            quiver.set_facecolors(colors)

        return quiver

    @property
    def cfg(self) -> Dict:
        return self._cfg


def attach_polygons(polygon_1, polygon_2):
    x_a_1 = np.array(polygon_1[-1])
    x_b_1 = np.array(polygon_1[-2])
    x_a_2 = np.array(polygon_2[0])
    x_b_2 = np.array(polygon_2[1])

    x_ab_1 = x_a_1 - x_b_1
    x_ab_2 = x_a_2 - x_b_2
    theta_1 = np.arctan2(x_ab_1[0], x_ab_1[1])
    theta_2 = np.arctan2(x_ab_2[0], x_ab_2[1])
    theta = theta_2 - theta_1
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    s = np.linalg.norm(x_ab_1) / np.linalg.norm(x_ab_2)

    x_offset = x_a_1 - x_a_2

    new_polygon_2 = polygon_2.copy()
    for i, x in enumerate(new_polygon_2):
        x = np.array(x)
        # new_x = s * (R @ (x - x_offset) ) #+ x_ref)
        # new_x = (x + x_offset) # R @ (x + x_offset)
        new_x = s * R @ (x - x_a_2) + x_a_1
        new_polygon_2[i] = tuple(new_x)

    return new_polygon_2
