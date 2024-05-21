#! /usr/bin/env python3
# Author : Yanwei Wang
# Email  : felixw@mit.edu
#
# Distributed under terms of the MIT license.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle

def load_drawn(dir_path):
    with open(os.path.join(dir_path, 'task.pkl'), 'rb') as inp:
        task = pickle.load(inp)
    data = pd.read_csv(os.path.join(dir_path, 'data.csv')) # drawn data
    df = data[['traj_id', 't', 'x', 'y']]
    return task, df

def load_perturbed(dir_path, succ_file='succ_traj.npy', fail_file='fail_traj.npy'):
    succ_data = np.swapaxes(np.load(os.path.join(dir_path, succ_file)), 1, 2)
    fail_data = np.swapaxes(np.load(os.path.join(dir_path, fail_file)), 1, 2)
    return succ_data, fail_data

def visualize_drawn(task, df, fig_size=(6, 6)):
    fig, ax = task.build_map(fig_size=fig_size)
    sns.scatterplot(data=df, x='x', y='y', hue='traj_id', ax=ax, s=10)
    ax.set_title('Successful Demonstrations in the Task Environment')

def perturb_traj(orig, task):
    # Symmetrical continous Gaussian perturbation
    impulse_start = random.randint(0, len(orig)-2)
    impulse_end = random.randint(impulse_start+1, len(orig)-1)
    impulse_mean = (impulse_start + impulse_end)/2
    impulse_target_x = random.uniform(-8, 8)
    impulse_target_y = random.uniform(-8, 8)
    max_relative_dist = 5 # np.exp(-5) ~= 0.006

    kernel = np.exp(-max_relative_dist*(np.array(range(len(orig))) - impulse_mean)**2 / ((impulse_start-impulse_mean)**2))
    perturbed = orig.copy()
    perturbed[:, 1] += (impulse_target_y-perturbed[:, 1])*kernel
    perturbed[:, 0] += (impulse_target_x-perturbed[:, 0])*kernel

    succ_traj, _, _ = task.validify_traj(perturbed.T)

    return perturbed, succ_traj #, end_in_red

def plot_traj(orig, perturbed):
    fig, ax = plt.subplots()
    rect1 = matplotlib.patches.Rectangle((-5, -2),
                                           5, 4,
                                         color ='yellow', alpha=0.2)

    rect2 = matplotlib.patches.Rectangle((0, -2),
                                         5, 4,
                                         color ='pink', alpha=0.2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.plot(orig[:, 0], orig[:, 1], label='Orig')
    ax.plot(perturbed[:, 0], perturbed[:, 1], label='Perturbed')
    plt.scatter(perturbed[:, 0], perturbed[:, 1], s=10, alpha=0.5)
    ax.set(ylim=(-8, 8))
    ax.set(xlim=(-8, 8))

def plot_batch(task, perturbed, fig_size=(6, 6), fig_ax=None, title=None):
    fig, ax = task.build_map(fig_ax=fig_ax)

    for traj in perturbed:
        ax.scatter(traj[:, 0], traj[:, 1], alpha=0.5, s=2)
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.2)

    ax.set(ylim=(-8, 8))
    ax.set(xlim=(-8, 8))

    if title is not None:
        ax.set_title(title)

def generate_counter_factual(task, traj_generator, evaluator, num_traj=50):
    fail_traj = []
    while True:
        if len(fail_traj) >= num_traj:
            break
        traj = traj_generator(evaluator)
        is_succ, _, _ = task.validify_traj(traj.T)
        if not is_succ:
            fail_traj.append(traj)
    return fail_traj

def generate_perturbed(task, num_traj=500):
    df = task.data[['traj_id','t', 'x', 'y']]

    # gather traj
    succ_traj = []
    while True:
        if len(succ_traj) >= num_traj:
            break
        traj_idx = random.randint(df['traj_id'].min(), df['traj_id'].max())
        traj = df.loc[df['traj_id']==traj_idx]
        orig = traj[['x', 'y']].to_numpy()
        perturbed, is_succ = perturb_traj(orig, task)
        if is_succ:
            succ_traj.append(perturbed)

    fail_traj = []
    while True:
        if len(fail_traj) >= num_traj:
            break
        traj_idx = random.randint(df['traj_id'].min(), df['traj_id'].max())
        traj = df.loc[df['traj_id']==traj_idx]
        orig = traj[['x', 'y']].to_numpy()
        perturbed, is_succ = perturb_traj(orig, task)
        # if not is_succ and end_in_red:
        if not is_succ:
            fail_traj.append(perturbed)

    return succ_traj, fail_traj


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--num", type=int, default=500)
    args = parser.parse_args()

    with open(os.path.join(args.dir, 'task.pkl'), 'rb') as inp:
        task = pickle.load(inp)

    succ_traj, fail_traj = generate_perturbed(task, num_traj=args.num)

    succ_data = np.swapaxes(np.stack(succ_traj), 1, 2)
    np.save(os.path.join(args.dir, 'succ_traj.npy'), succ_data.astype('float32'))
    fail_data = np.swapaxes(np.stack(fail_traj), 1, 2)
    np.save(os.path.join(args.dir, 'fail_traj.npy'), fail_data.astype('float32'))

    plot_batch(task, succ_traj, title='Successful Perturbed Trajectories')
    plot_batch(task, fail_traj, title='Failed Perturbed Trajectories')
    plt.show()


