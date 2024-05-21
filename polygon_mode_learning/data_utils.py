#! /usr/bin/env python3
# Author : Tsun-Hsuan Wang
#
# Distributed under terms of the MIT license.

from typing import Sequence, Tuple, Dict, Union, Any
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from polygon_mode_learning.data_types import Mode, Transition, Region


def taskdef2graph(modes: Sequence[Mode], transitions: Sequence[Transition]) -> nx.Graph:
    """Convert a list of modes and transitions to a directed graph."""
    G = nx.Graph().to_directed()
    for m in modes:
        G.add_node(m.name)

    for t in transitions:
        nx.add_path(G, [t.inp.name, t.out.name])

    return G


def plot_graph(
    G: nx.Graph,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    fpath: str = None,
    show: bool = False,
):
    """Visualize a networkx graph."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    nx.draw(G, ax=ax, with_labels=True)

    if fpath is not None:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        fig.savefig(fpath)
        plt.close(fig)

    if show:
        plt.show()


def resample_traj(traj: np.ndarray, n_pts: int) -> np.ndarray:
    """Resample along the trajectory to get `n_pts` points.
       `traj` is of shape (D, N) where N is trajectory length."""
    prog = np.linalg.norm(traj[:, 1:] - traj[:, :-1], axis=0)
    prog = np.insert(prog, 0, 0.)
    prog = np.cumsum(prog)
    prog = prog / prog.max()

    prog_resampled = np.linspace(0., 1., n_pts)
    resampled_traj = []
    for traj_i in traj:
        f = interp1d(prog, traj_i)
        resampled_traj_i = f(prog_resampled)
        resampled_traj.append(resampled_traj_i)
    resampled_traj = np.stack(resampled_traj, axis=0)

    return resampled_traj


def xy_in_region(xy: Sequence[float], region: Region) -> bool:
    """Check whether a coordinate is within the region."""
    if region.polygon is not None:
        from shapely import Polygon, Point
        polygon = Polygon(region.polygon)
        pt = Point(xy)
        return polygon.covers(pt)
    else:
        region_s = np.array(region.xy)
        region_e = region_s + np.array([region.width, region.height])
        return np.all(xy >= region_s) and np.all(xy <= region_e)



def traj2modeseq(traj: np.ndarray, modes: Sequence[Mode]) -> Sequence[Mode]:
    """Convert a trajectory into a sequence of modes.
       `traj` is of shape (D, N) where N is trajectory length."""
    mode_seq = []
    for i in range(traj.shape[1]):
        xy = traj[:, i]
        has_matched_mode = False
        has_free_mode = False
        for mode in modes:
            if mode.region is None: # NOTE: assume there is only one free mode
                free_mode = mode
                has_free_mode = True
            else:
                if xy_in_region(xy, mode.region):
                    xy_mode = mode
                    has_matched_mode = True
                    break
        if not has_matched_mode:
            if has_free_mode:
                xy_mode = free_mode
            else:
                raise ValueError(f"No matched mode at {xy}")
        mode_seq.append(xy_mode)

    return mode_seq


def validify_transitions(
    modes: Sequence[Mode],
    transitions: Sequence[Transition],
    required_paths: Sequence[Mode],
) -> Tuple[bool, np.ndarray]:
    """Check if a mode sequence obeys the given set of transitions.
       Return whether the mode sequence is valid and which transition is obeyed for every timestep."""
    # Check validity of transition
    valid_transitions = []
    for i in range(len(modes) - 1):
        curr_mode = modes[i]
        next_mode = modes[i + 1]
        v_t = []
        for t in transitions:
            is_valid = (curr_mode.id == t.inp.id) and (next_mode.id == t.out.id)
            v_t.append(is_valid)
        valid_transitions.append(v_t)
    valid_transitions = np.array(valid_transitions)

    all_valid = np.all(valid_transitions.sum(1) > 0)

    # Check initial state is valid
    valid_initial = modes[0].is_initial

    # Check if required paths are traced out
    reduced_modes = []
    for m in modes:
        if len(reduced_modes) == 0 or m != reduced_modes[-1]:
            reduced_modes.append(m)

    satisfy_requirement = True
    for path in required_paths:
        relevant_modes = [m for m in reduced_modes if m in path]
        pattern_matched = False
        for i in range(len(relevant_modes) - len(path) + 1):
            if path == relevant_modes[i:i+len(path)]:
                pattern_matched = True
        satisfy_requirement = satisfy_requirement and pattern_matched

    # Summarize trajectory-level success
    success = all_valid and valid_initial and satisfy_requirement

    return success, valid_transitions


def arrow_handler_map():
    """Create a handler map for plotting, e.g., using arrow in legend.
       Ref: https://stackoverflow.com/questions/22348229/matplotlib-legend-for-an-arrow"""
    def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
        return p

    return {mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)}


def seq2df(data: Dict[str, Union[int, float, np.ndarray]]) -> pd.DataFrame:
    """Convert sequence data to data frame with `t` as element identifier."""
    seq_data = []
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            seq_data.append(v)
    seq_lens = [v.shape[0] for v in seq_data]
    assert len(np.unique(seq_lens)) == 1

    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            data[k] = np.array([v] * seq_lens[0])

    data["t"] = np.arange(seq_lens[0])

    return pd.DataFrame(data)


def sublist(ls1: Sequence[Any], ls2: Sequence[Any]):
    """Check if `ls2` is a sublist of `ls1`.
    Example usage:
        >>> sublist([], [1,2,3])
        True
        >>> sublist([1,2,3,4], [2,5,3])
        True
        >>> sublist([1,2,3,4], [0,3,2])
        False
        >>> sublist([1,2,3,4], [1,2,5,6,7,8,5,76,4,3])
        False
    """
    def get_all_in(one, another):
        for element in one:
            if element in another:
                yield element

    for x1, x2 in zip(get_all_in(ls1, ls2), get_all_in(ls2, ls1)):
        if x1 != x2:
            return False

    return True
