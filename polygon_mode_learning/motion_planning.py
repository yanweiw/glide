#! /usr/bin/env python3
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
#
# Distributed under terms of the MIT license.

from matplotlib import pyplot as plt
from typing import Tuple
from PIL import Image
import numpy as np
import imageio
import torch
import os


Vector2f = Tuple[float, float]

def rrt_plan(pos: Vector2f, goal: Vector2f, obstacles: np.ndarray):
    class RRTNode():
        def __init__(self, pos, parent=None):
            self.pos = pos
            self.parent = parent

    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def sample():
        if np.random.rand() < 0.05:
            return goal
        else:
            return np.random.uniform(-8, 8, 2)

    def nearest_neighbor(nodes, q):
        return min(nodes, key=lambda n: dist(n.pos, q))

    def steer(q, q_near, epsilon=0.1):
        if dist(q, q_near) < epsilon:
            return q
        else:
            return q_near + epsilon * (q - q_near) / dist(q, q_near)

    def is_collision_free(q1, q2, obstacles):
        # generate a linear path with 20 points
        path = np.linspace(q1, q2, 20)
        for p in path:
            if not is_collision_free_point(p, obstacles):
                return False
        return True

    def is_collision_free_point(q, obstacles):
        # obstacles is N x 2
        # q is 2
        q = np.array(q)
        distances = np.linalg.norm(obstacles - q, axis=1)
        return np.all(distances > 0.4)

    obstacles = np.array(obstacles)
    root = RRTNode(pos)
    nodes = [root]
    for i in range(1000):
        q = sample()
        q_near_node = nearest_neighbor(nodes, q)
        q_new = steer(q, q_near_node.pos)
        if is_collision_free_point(q_new, obstacles):
            q_new_node = RRTNode(q_new, q_near_node)
            nodes.append(q_new_node)
            if dist(q_new, goal) < 0.1:
                # print('Found path!')
                break
    else:
        # print('No path found!')
        return None

    path = []
    node = nodes[-1]
    while node is not None:
        path.append(node.pos)
        node = node.parent
    path = np.array(path[::-1])
    return path


def potential_field(pos: Vector2f, goal: Vector2f, obstacles: np.ndarray, k_att: float = 1, k_rep: float = 10.0, d_0: float = 0.8):
    pos = np.array(pos)
    goal = np.array(goal)
    obstacles = np.array(obstacles)

    U_att = -0.5 * k_att * np.linalg.norm(pos - goal) ** 2
    grad_U_att = -k_att * (pos - goal)

    U_rep = 0
    grad_U_rep = np.zeros_like(pos)

    dist = np.clip(np.linalg.norm(pos - obstacles, axis=-1), 1e-6, None)
    mask = dist < d_0
    U_rep += ((0.5 * k_rep * (1.0 / dist - 1.0 / d_0) ** 2) * mask).sum()
    grad_U_rep += (k_rep * ((1.0 / dist - 1.0 / d_0) / dist ** 2)[:, None] * (pos - obstacles) * mask[:, None]).sum(axis=0)

    U = U_rep + U_att
    grad_U = grad_U_att + grad_U_rep
    grad_U = grad_U / np.linalg.norm(grad_U) * 0.1
    return U, grad_U


def compute_waypoints(traj_data, net):
    trajectories = torch.tensor(traj_data).float() 
    with torch.no_grad():
        traj_mode, _ = net.pred_mode(trajectories)

    nr_trajectories, nr_points, _ = trajectories.shape
    traj_mode = traj_mode.reshape(nr_trajectories, nr_points, net.num_guess, net.num_modes)
    traj_mode = traj_mode[..., net.best_guess_idx, :]
    traj_mode = traj_mode.argmax(-1)

    waypoints = []
    for i in range(nr_trajectories):
        waypoints.append([])
        for j in range(1, net.num_modes):
            idx = (traj_mode[i] == j).nonzero().reshape(-1)
            if len(idx) > 0:
                idx = idx[0]
                waypoints[-1].append(trajectories[i, idx].cpu().numpy())
        if len(waypoints[-1]) != net.num_modes - 1:
            # delete
            del waypoints[-1]

    average_waypoints = []
    for i in range(1, net.num_modes):
        all_waypoints = [x[i - 1] for x in waypoints]
        average_waypoints.append(np.mean(all_waypoints, axis=0) if len(all_waypoints) > 0 else None)

    return average_waypoints
 

def visualize_waypoints(waypoints, net, mode_color, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(mode_color.detach().cpu().numpy(), extent=[-8, 8, -8, 8], alpha=0.2, origin='lower')
    for i in range(1, net.num_modes):
        if waypoints[i - 1] is not None:
            ax.scatter(waypoints[i - 1][0], waypoints[i - 1][1], color=net.mode_colors[i], s=10)
    ax.set_title('Average Waypoints')


def compute_potential_flow(mode_belief_argmax, average_waypoints, net):
    x_size, y_size = mode_belief_argmax.shape
    def _to_state_xy_1(x): return x * (16 / (x_size-1)) - 8
    def _to_state_xy(x, y): return np.array([x * (16 / (x_size-1)) - 8, y * (16 / (y_size-1)) - 8])

    obstacle_map = (mode_belief_argmax > 1)
    obstacle_y, obstacle_x = np.where(obstacle_map == 1)
    obstacles = np.stack([_to_state_xy_1(obstacle_x), _to_state_xy_1(obstacle_y)], axis=1)

    grid_potential = np.zeros((mode_belief_argmax.shape[0], mode_belief_argmax.shape[1]))
    grid_action = np.zeros((mode_belief_argmax.shape[0], mode_belief_argmax.shape[1], 2))
    for i in range(net.num_modes):
        if i == 0:
            all_points_in_mode = np.where(mode_belief_argmax == i)
            goal = average_waypoints[0]
            for y, x in zip(*all_points_in_mode):
                pos = _to_state_xy_1(x), _to_state_xy_1(y)
                U, grad_U = potential_field(pos, goal, obstacles)
                grid_action[y, x] = grad_U
                grid_potential[y, x] = U
        elif i == len(average_waypoints):
            all_points_in_mode = np.where(mode_belief_argmax == i)
            center = (np.mean(all_points_in_mode[1]), np.mean(all_points_in_mode[0]))
            for y, x in zip(*all_points_in_mode):
                action = center - np.array([x, y])
                action = action / np.linalg.norm(action) * 0.1
                grid_action[y, x] = action
        else:
            next_waypoint = average_waypoints[i]
            for y, x in zip(*np.where(mode_belief_argmax == i)):
                action = next_waypoint - _to_state_xy(x, y)
                action = action / np.linalg.norm(action) * 0.1
                grid_action[y, x] = action
    print('Potential field computation done (for visualization only)')
    return grid_action, obstacles

def visualize_potential_flow(mode_belief_argmax, grid_action, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    x_size, y_size = mode_belief_argmax.shape
    for y, x in zip(*np.where(mode_belief_argmax > -1)):
        if y % 4 == 0 and x % 4 == 0:
            xx = x * (16 / (x_size-1)) - 8
            yy = y * (16 / (y_size-1)) - 8
            ax.arrow(xx, yy, grid_action[y, x, 0], grid_action[y, x, 1], head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.set_title('Potential Field')


class TrajectorySaver(object):
    def __init__(self, env, save_dir='figs', enable=False):
        self.env = env
        self.enable = enable
        if self.enable:
            plt.figure(figsize=(8, 8))
            env.fig = plt.gcf()
            env.ax = plt.gca()
            self.images = list()
            self.tmp_dir = save_dir
            self.tmp_file = os.path.join(self.tmp_dir, 'state.jpg')
            self.tmp_video_file = os.path.join(self.tmp_dir, 'trajectory.mp4')

    def save_fig(self, failed=False):  # Helper function for saving the videos.
        fig = self.env.fig
        ax = self.env.ax

        if failed:
            plt.title(f'Step={self.env.step_count} (Failed - Crossing Invalid Boundaries)', fontsize=20)
        plt.draw()

        fig.savefig(self.tmp_file)
        im = Image.open(self.tmp_file)
        im = np.array(im)
        if self.env.last_perturb_config is None:
            if failed:
                return [im] * 45
            return [im]
        else:
            self.env.last_perturb_config = None
            return [im] * 20

    def tick(self, done=False, reward=0):
        failed = done and reward < 1
        if self.enable:
            self.env.render_inner(False, title_state=True)
            self.images.extend(self.save_fig(failed=failed))

    def finalize(self):
        if self.enable:
            imageio.mimsave(self.tmp_video_file, self.images, fps=45, codec='h264')
            plt.close(self.env.fig)
            self.env.fig = None
            self.env.ax = None
            self.images = None
            return self.tmp_video_file
        return None


def run_episode(env, grid_action, grid_mode_argmax, average_waypoints, obstacles, save_video_dir=None):
    if save_video_dir is not None:
        save_video = True
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
    else:
        save_video = False

    state = env.reset(perturb_quota=3)  # Allow at most 3 perturbations.
    video_saver = TrajectorySaver(env, save_dir=save_video_dir, enable=save_video)
    video_saver.tick()
    x = np.linspace(-8, 8, 200)
    y = np.linspace(-8, 8, 200)

    for j in range(1000):
        cur_x, cur_y = state

        # find the closest point
        cur_x_idx = np.argmin(np.abs(x - cur_x))
        cur_y_idx = np.argmin(np.abs(y - cur_y))
        action_index = grid_action[cur_y_idx, cur_x_idx]

        mode = grid_mode_argmax[cur_y_idx, cur_x_idx]
        action = None
        if mode == 0:
            path = rrt_plan(state, average_waypoints[0], obstacles)
            if path is None:
                action = np.array([action_index[0], action_index[1]])
            else:
                for k in range(1, len(path)):
                    action = path[k] - path[k - 1]
                    state, reward, done, info = env.step(action)
                    video_saver.tick(done, reward)

                    # if distance > 0.1 then we break
                    if np.linalg.norm(np.asarray(path[k]) - np.asarray(state)) > 0.1:
                        action = None
                        break
                    if done:
                        if reward > 1:
                            return True, video_saver
                        return False, video_saver
        else:
            action = np.array([action_index[0], action_index[1]])

        if action is not None:
            state, reward, done, _ = env.step(action)
            video_saver.tick(done, reward)
        if done:
            if reward > 1:
                return True, video_saver
            return False, video_saver

    return False, video_saver