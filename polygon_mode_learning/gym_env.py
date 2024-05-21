#! /usr/bin/env python3
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
#
# Distributed under terms of the MIT license.

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from polygon_mode_learning.two_polygons import TwoPolygons


class Polygon2DEnv(object):
    def __init__(self, two_polygons: TwoPolygons, max_move_distance=0.1, start_from_free=False):
        self._two_polygons = two_polygons
        self._modes = two_polygons._modes
        self._max_move_distance = max_move_distance
        self._start_from_free = start_from_free

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        self._max_episode_steps = 500

        self.x, self.y = (0, 0)
        self.current_mode = None
        self.visited_modes = list()
        self.step_count = 0
        self.perturb_quota = 0
        self.last_perturb = None
        self.last_perturb_config = None

    @property
    def modes(self):
        return self._modes

    @property
    def initial_mode(self):
        assert len(self._modes) >= 2
        assert self._modes[1].is_initial
        return self._modes[1]

    def get_mode(self, state):
        x, y = state
        mode_idx = 0
        for i, mode in enumerate(self._modes):
            if i == 0:
                continue
            if mode.region.in_region(x, y):
                mode_idx = i
                break
        return mode_idx

    def reset(self, perturb_quota=0):
        if not self._start_from_free:
            mode = self.initial_mode
            self.x, self.y = mode.region.sample()
            self.current_mode = mode
            self.visited_modes = [self._modes[0], mode]
        else:
            mode = self._modes[0]
            while True:
                x = np.random.uniform(self._two_polygons.XLIM[0], self._two_polygons.XLIM[1])
                y = np.random.uniform(self._two_polygons.YLIM[0], self._two_polygons.YLIM[1])
                if not any([mode.region.in_region(x, y) for mode in self._modes[1:]]):
                    break
            self.x, self.y = x, y
            self.current_mode = mode
            self.visited_modes = [mode]
        self.perturb_quota = perturb_quota
        self.last_perturb = None
        self.last_perturb_config = None
        self.step_count = 0
        return self.x, self.y

    def step(self, action):
        self.step_count += 1
        dx, dy = action

        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm > self._max_move_distance:
            dx *= self._max_move_distance / norm
            dy *= self._max_move_distance / norm

        # move the point
        new_x = self.x + dx
        new_y = self.y + dy

        # check the mode of the new point
        new_mode = self._modes[0]
        for mode in self._modes[1:]:
            if mode.region.in_region(new_x, new_y):
                new_mode = mode
                break

        if mode == self.current_mode:
            reward = 0
        else:
            reward = -1
            for transition in self._two_polygons._transitions:
                if transition.inp == self.current_mode and transition.out == new_mode:
                    reward = 1 if new_mode not in self.visited_modes else 0
                    break

        done = False
        if reward == -1:
            done = True
        if reward == 1 and new_mode.is_goal:
            reward = 3
            done = True

        self.x = new_x
        self.y = new_y
        self.current_mode = new_mode
        if new_mode not in self.visited_modes:
            self.visited_modes.append(new_mode)

        if not done:
            p = 0.03
            if self.perturb_quota > 0:
                allow_perturb = len(self.visited_modes) > 1 and (self.last_perturb is None or self.step_count - self.last_perturb > 100)

                # if self.perturb_quota == 1:
                #     allow_perturb = len(self.visited_modes) > 3

                if allow_perturb and np.random.uniform() < p:
                    self.perturb_quota -= 1
                    self.last_perturb = self.step_count
                    mode = self._modes[0]
                    while True:
                        x = np.random.uniform(self._two_polygons.XLIM[0], self._two_polygons.XLIM[1])
                        y = np.random.uniform(self._two_polygons.YLIM[0], self._two_polygons.YLIM[1])
                        if not any([mode.region.in_region(x, y) for mode in self._modes[1:]]):
                            break
                    previous_xy = self.x, self.y
                    self.x, self.y = x, y
                    self.last_perturb_config = (previous_xy, (x, y))
                else:
                    self.last_perturb_config = None
            else:
                self.last_perturb_config = None

        return (self.x, self.y), reward, done, None

    def render_human_play(self):
        plt.figure(figsize=(8, 8))
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.render_inner()

    def render_policy(self, policy_fn):
        plt.figure(figsize=(8, 8))
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.policy_fn = policy_fn
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press_policy)
        self.render_inner()

    def render_policy_trajectory(self, policy_fn, num_steps=500, perturb_quota=0):
        state = self.reset(perturb_quota=perturb_quota)
        states = [state]
        for _ in range(num_steps):
            action = policy_fn(state)
            state, reward, done, _ = self.step(action)
            states.append(state)
            if done:
                break

        plt.figure(figsize=(8, 8))
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.render_inner(wait_key=False)
        for i in range(len(states) - 1):
            self.ax.plot([states[i][0], states[i + 1][0]], [states[i][1], states[i + 1][1]], color='red')

        key = None

        def keypress(event):
            nonlocal key
            key = event.key
        self.fig.canvas.mpl_connect('key_press_event', keypress)

        success = self.current_mode.is_goal
        print('success = {}'.format(success))
        plt.waitforbuttonpress()
        plt.close()
        return success, key

    def render_policy_action_field(self, batch_policy_fn, grid_modes=None, resolution_n=50, show=True, save=None):
        xs = np.linspace(self._two_polygons.XLIM[0], self._two_polygons.XLIM[1], resolution_n)
        ys = np.linspace(self._two_polygons.YLIM[0], self._two_polygons.YLIM[1], resolution_n)

        xs, ys = np.meshgrid(xs, ys)
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        states = np.stack([xs, ys], axis=1)
        actions = batch_policy_fn(states)

        plt.figure(figsize=(8, 8))
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.render_inner(wait_key=False, render_current=False)

        # render the action field as arrows
        for i in range(len(xs)):
            if grid_modes is not None:
                x_index = int((xs[i] - self._two_polygons.XLIM[0]) / (self._two_polygons.XLIM[1] - self._two_polygons.XLIM[0]) * 199)
                y_index = int((ys[i] - self._two_polygons.YLIM[0]) / (self._two_polygons.YLIM[1] - self._two_polygons.YLIM[0]) * 199)
                mode = grid_modes[y_index, x_index]
                mode_color = self._modes[mode].color
                self.ax.arrow(xs[i], ys[i], actions[i][0], actions[i][1], head_width=0.05, head_length=0.1, color=mode_color)
            else:
                self.ax.arrow(xs[i], ys[i], actions[i][0], actions[i][1], head_width=0.05, head_length=0.1, fc='k', ec='k')

        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)

    def render_data(self, list_of_sa, show=True, save=None):
        plt.figure(figsize=(8, 8))
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.render_inner(wait_key=False, render_current=False)

        random_indices = np.random.choice(len(list_of_sa), 1000, replace=False)
        for state, action in [list_of_sa[i] for i in random_indices]:
            action = np.array(action)
            action = action / np.linalg.norm(action) * 0.1
            mode = state[2:].argmax()
            mode_color = self._modes[mode].color
            if mode_color is not None:
                self.ax.arrow(state[0], state[1], action[0], action[1], head_width=0.05, head_length=0.1, color=mode_color)
            else:
                self.ax.arrow(state[0], state[1], action[0], action[1], head_width=0.05, head_length=0.1, fc='k', ec='k')

        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)

    def render_inner(self, wait_key=True, render_current=True, title_state=False):
        # clear the ax
        self.ax.clear()

        for mode in self._modes:
            region = mode.region
            if region is not None:
                self.ax.add_patch(matplotlib.patches.Polygon(region.polygon, color=mode.color, alpha=0.5))

        # add the current position
        # print('draw point ({}, {})'.format(self.x, self.y))
        if render_current:
            self.ax.add_patch(matplotlib.patches.Circle((self.x, self.y), radius=0.1, color='red'))

        plt.xlim(self._two_polygons.XLIM)
        plt.ylim(self._two_polygons.YLIM)

        if self.last_perturb_config is not None:
            previous_xy = self.last_perturb_config[0]
            next_xy = self.last_perturb_config[1]
            # draw a line
            self.ax.plot([previous_xy[0], next_xy[0]], [previous_xy[1], next_xy[1]], color='red')

        if title_state:
            plt.axis('off')
            # make the title larger
            plt.title(f'Step={self.step_count} Mode={self.current_mode.name}', fontsize=20)
            plt.tight_layout()

        plt.draw()
        if wait_key:
            if self.current_mode.is_goal:
                return
            plt.waitforbuttonpress(0)

    # add a keyboard listener
    def on_key_press(self, event):
        if event.key == 'escape':
            plt.close()
            return
        if event.key == 'left':
            action = (-0.1, 0)
        elif event.key == 'right':
            action = (0.1, 0)
        elif event.key == 'up':
            action = (0, 0.1)
        elif event.key == 'down':
            action = (0, -0.1)
        else:
            return

        state, reward, done, _ = self.step(action)
        print('state = {}, reward = {}, done = {}'.format(state, reward, done))
        self.render_inner()

    def on_key_press_policy(self, event):
        print('key pressed', event.key)
        if event.key == 'escape':
            plt.close()
            return
        if event.key == ' ':
            for i in range(5):
                action = self.policy_fn((self.x, self.y))
                state, reward, done, _ = self.step(action)
                print('state = {}, reward = {}, done = {}'.format(state, reward, done))
                if done:
                    break
            self.render_inner()


if __name__ == '__main__':
    import pickle
    import matplotlib
    import matplotlib.pyplot as plt

    with open('./tmp1/task.pkl', 'rb') as f:
        task = pickle.load(f)

    env = Polygon2DEnv(task)
    env.reset()
    env.render_human_play()

