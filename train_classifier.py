#! /usr/bin/env python3
# Author : Yanwei Wang
# Email  : felixw@mit.edu
#
# Distributed under terms of the MIT license.

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import time 


class Net(nn.Module):
    """The mode classification network."""

    def __init__(self, num_layers, num_modes, feas_mat=None, num_guess=10, traj_len=100):
        """Initialize the network.

        Args:
            num_layers: the number of layers in the network.
            num_modes: the number of modes to predict.
            feas_mat: the feasibility matrix.
            num_guess: the number of guesses. Specifically, we will initialize `num_guess` branches in the network for mode prediction.
        """        
        super().__init__()

        # mode prediction
        assert num_layers >= 2
        self.layers = [nn.Linear(2, 128)]
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(128, 128))
        self.layers = nn.ModuleList(self.layers)
        self.predit = nn.Linear(128, num_modes * num_guess) # predict multiple guesses in parallel helps optimization to converge faster

        self.dropout = nn.Dropout(p=0.0)
        self.num_modes = num_modes
        self.num_guess = num_guess
        self.mode_colors = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6]
            ]
        self.best_guess_idx = None
        self.traj_len = traj_len
        # feasibility matrix
        if feas_mat is not None:
            self.register_buffer('feasible', torch.tensor(feas_mat).float())
            assert self.feasible.shape == (num_modes, num_modes)

    def pred_mode(self, x):
        """Predict the mode.

        Args:
            x: the input trajectory, of shape [T, 2]. We typically use this function for T = 1.

        Returns:
            mode: the predicted mode, of shape [T * num_guess, num_modes]
            mode_log: the log of the predicted mode, of shape [T * num_guess, num_modes]
        """        
        out = x.reshape(-1, 2)
        for layer in self.layers:
            out = self.dropout(F.relu(layer(out)))
        out = self.predit(out)
        out = out.reshape(-1, self.num_modes)
        temperature = 1
        mode = nn.Softmax(dim=1)(out / temperature)
        mode_log = nn.LogSoftmax(dim=1)(out / temperature)
        return mode, mode_log

    def forward(self, x):
        mode, mode_log = self.pred_mode(x)
        mode = mode.reshape(-1, self.traj_len, self.num_guess, self.num_modes)
        mode_log = mode_log.reshape(-1, self.traj_len, self.num_guess, self.num_modes)
        final_mode = mode_log[:, -1, :, :]
        init_mode = mode_log[:, 0, :, :]
        next_mode = torch.clone(mode)[:, 1:, :, :]
        curr_mode = mode[:, :self.traj_len-1, :, :]
        assert curr_mode.shape == next_mode.shape
        curr_feas = torch.matmul(curr_mode, self.feasible)
        curr_feas = curr_feas[:, :, :, None, :]
        next_mode = next_mode[:, :, :, :, None]
        feasibility = torch.matmul(curr_feas, next_mode).squeeze(-1).squeeze(-1)
        assert feasibility.shape[1:] == (self.traj_len-1, self.num_guess)
        assert final_mode.shape[1:] == (self.num_guess, self.num_modes)
        assert init_mode.shape[1:] == (self.num_guess, self.num_modes)
        return feasibility, final_mode, init_mode


def evaluate(net, wandb, save_dir, device):
    # eval by plotting mode sections
    x = np.linspace(-8, 8, 200)
    y = np.linspace(-8, 8, 200)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.flatten(), Y.flatten()]).T
    XY = torch.from_numpy(XY).float()

    with torch.no_grad():
        mode, mode_log = net.pred_mode(XY.to(device))
    mode = mode.reshape(200, 200, net.num_guess, net.num_modes)[:, :, net.best_guess_idx, :]

    color_matrix = torch.tensor(net.mode_colors)[:wandb.config.num_modes, :]
    colored_mode = torch.matmul(mode, color_matrix.to(device))
    colored_mode = torch.clamp(colored_mode, min=0, max=1)
    assert colored_mode.shape == (200, 200, 3)

    color_matrix = torch.tensor(net.mode_colors)[:wandb.config.num_modes, :].to(device)
    mode_argmax = mode.argmax(-1)
    mode_argmax = mode_argmax.reshape(200, 200)
    mode_argmax_one_hot = torch.zeros(200, 200, wandb.config.num_modes).to(device)
    mode_argmax_one_hot.scatter_(2, mode_argmax[:, :, None], 1)
    colored_mode_argmax = torch.matmul(mode_argmax_one_hot, color_matrix)
    colored_mode_argmax = torch.clamp(colored_mode_argmax, min=0, max=1)

    fig, ax = plt.subplots()
    plt.scatter(X.flatten(), Y.flatten(), c=colored_mode.cpu().numpy().reshape(-1, 3), s=1)
    plt.savefig(os.path.join(save_dir, 'modes.png'))
    plt.close()

    fig, ax = plt.subplots()
    plt.scatter(X.flatten(), Y.flatten(), c=colored_mode_argmax.cpu().numpy().reshape(-1, 3), s=1)
    plt.savefig(os.path.join(save_dir, 'modes_argmax.png'))
    plt.close()

    fig, ax = plt.subplots()
    plt.imshow(colored_mode.detach().cpu().numpy()[::-1], extent=[-8, 8, -8, 8], alpha=0.5, origin='upper')
    plt.savefig(os.path.join(save_dir, 'modes_imshow.png'))
    wandb.log({'valid/': plt})
    plt.close()


def visualize_learned_modes(env, net):
    x = np.linspace(-8, 8, 200)
    y = np.linspace(-8, 8, 200)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.flatten(), Y.flatten()]).T
    XY = torch.from_numpy(XY).float()

    with torch.no_grad():
        mode_all_guess, _ = net.pred_mode(XY)

    gt_mode = torch.zeros(200, 200)
    for i in range(200):
        for j in range(200):
            gt_mode[i, j] = env.get_mode([X[i, j], Y[i, j]])

    color_matrix = torch.tensor(net.mode_colors)[:net.num_modes, :]

    # For all the guesses, visualize the mode.
    visualizations = list()
    for guess_idx in range(10):
        mode = mode_all_guess.reshape(200, 200, net.num_guess, net.num_modes)[:, :, guess_idx, :]
        colored_mode = torch.matmul(mode, color_matrix)
        colored_mode = torch.clamp(colored_mode, min=0, max=1)
        color_matrix = torch.tensor(net.mode_colors)[:net.num_modes, :]
        visualizations.append(colored_mode.detach().numpy())

        mode_argmax = mode.argmax(-1)
        mode_argmax = mode_argmax.reshape(200, 200)
        # compute the mode accuracy
        mode_acc = (mode_argmax == gt_mode).float().mean()
        print('Guess idx:', guess_idx, 'Overlap with Ground Truth Mode:', mode_acc.item())

    # Draw 2 rows and 5 columns of the mode predictions.
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(visualizations[i * 5 + j][::-1], alpha=0.2)  # Flip the y-axis to match the coordinate system.
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_title(f'Guess {i * 5 + j}')
    plt.tight_layout()
    plt.show()    
    return mode_all_guess, color_matrix


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--n_mds", type=int, required=True)
    args = parser.parse_args()
    nr_modes = args.n_mds
    print('Number of modes: ', nr_modes)

    wandb.init(
        mode="disabled",
        project="multiple_mode_learning",
        config={
            "optimizer": "ADAM",
            "lr": 0.001,
            "momentum": 0.99,
            "epochs": 100000,
            "num_layers": 2,
            "num_guess": 10,
            "traj_len": 100, 
            "feas_mat": np.array([
                [0.0,  0.0, -1.0, -2.0, -3.0],
                [0.0,  0.0,  0.0, -1.0, -2.0],
                [0.0,  0.0,  0.0,  0.0, -1.0],
                [0.0, -0.0,  0.0,  0.0, -0.0],
                [0.0, -0.0, -0.0,  0.0, -0.0]
            ])[:nr_modes, :nr_modes],
            "succ_data": os.path.join(args.dir, "succ_traj.npy"),
            "fail_data": os.path.join(args.dir, "fail_traj.npy"),
            "name": str(random.randint(0, 999999)).zfill(6),
            "succ_tran_weight": 1000,
            "fail_tran_weight": 1,
            "init_weight": 10,
            "num_modes": nr_modes,
        },
        save_code=True,
        tags=["fixed_feasibility", "multiple_mode_data"]
    )
    print('wandb run name: ', wandb.config.name)
    config = wandb.config

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    dataset = []
    for fname in [config.succ_data, config.fail_data]:
        data = np.load(fname)
        data = torch.tensor(data).float()  # shape (N, 2, len)
        data = torch.swapaxes(data, 1, 2)  # shape (N, len, 2)
        assert data.shape[1] == config.traj_len
        assert data.shape[2] == 2
        dataset.append(data.to(device))
        print(fname, ' has data shape: ', data.shape)

    net = Net(num_layers=config.num_layers, num_modes=config.num_modes, feas_mat=config.feas_mat, num_guess=config.num_guess, traj_len=config.traj_len)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)

    save_dir = 'weights/' + config.name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    losses = []
    start_time = time.time()
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        optimizer.zero_grad()
        # succ
        succ_feas, succ_final_mode, succ_init_mode = net(dataset[0])
        succ_loss = -succ_feas.mean([0, 1])                           # succ_feas shape [batch, len, num_guess]
        succ_final_mode = succ_final_mode.reshape(-1, net.num_modes)  # succ_final_mode shape [batch, num_guess, num_modes]
        succ_init_mode = succ_init_mode.reshape(-1, net.num_modes)    # succ_init_mode  shape [batch, num_guess, num_modes]
        succ_final_target = torch.ones(succ_final_mode.shape[0], dtype=torch.long) * (config.num_modes - 1)
        succ_init_target = torch.ones(succ_init_mode.shape[0], dtype=torch.long) * 0
        succ_final_loss = nn.NLLLoss(reduction='none')(succ_final_mode, succ_final_target.to(device)).reshape(-1, net.num_guess).mean(0)
        succ_init_loss = nn.NLLLoss(reduction='none')(succ_init_mode, succ_init_target.to(device)).reshape(-1, net.num_guess).mean(0)

        # fail
        fail_feas, fail_final_mode, fail_init_mode = net(dataset[1])
        fail_loss = torch.clamp(fail_feas.sum(1), min=-1).mean(0)
        fail_final_mode = fail_final_mode.reshape(-1, net.num_modes)  # fail_final_mode shape [batch, num_guess, num_modes]
        fail_init_mode = fail_init_mode.reshape(-1, net.num_modes)    # fail_init_mode  shape [batch, num_guess, num_modes]
        fail_final_target = torch.ones(fail_final_mode.shape[0], dtype=torch.long) * (config.num_modes - 1)
        fail_init_target = torch.ones(fail_init_mode.shape[0], dtype=torch.long) * 0
        fail_final_loss = nn.NLLLoss(reduction='none')(fail_final_mode, fail_final_target.to(device)).reshape(-1, net.num_guess).mean(0)
        fail_init_loss = nn.NLLLoss(reduction='none')(fail_init_mode, fail_init_target.to(device)).reshape(-1, net.num_guess).mean(0)

        assert succ_loss.shape == (net.num_guess,)
        assert fail_loss.shape == (net.num_guess,)
        assert succ_final_loss.shape == (net.num_guess,)
        assert fail_final_loss.shape == (net.num_guess,)
        assert succ_init_loss.shape == (net.num_guess,)
        assert fail_init_loss.shape == (net.num_guess,)

        succ_loss = config.succ_tran_weight * succ_loss
        fail_loss = config.fail_tran_weight * fail_loss
        init_loss = config.init_weight * (succ_init_loss + fail_init_loss)
        final_loss = config.init_weight * (succ_final_loss + fail_final_loss)

        loss = succ_loss + fail_loss + init_loss + final_loss
        net.best_guess_idx = torch.argmin(loss).detach()
        combined_loss = 0.1 * loss.mean() + 0.9 * loss[net.best_guess_idx]
        combined_loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            metrics = {
                "train/loss": combined_loss,
                "train/succ_loss": succ_loss[net.best_guess_idx],
                "train/fail_loss": fail_loss[net.best_guess_idx],
                "train/init_loss": init_loss[net.best_guess_idx],
                "train/final_loss": final_loss[net.best_guess_idx],
            }
            wandb.log({**metrics})

            min_elapsed = (time.time() - start_time) / 60
            print(f'[{epoch}, {min_elapsed:.1f} min] loss: {combined_loss:.3f}, succ: {succ_loss[net.best_guess_idx]:.3f}, fail: {fail_loss[net.best_guess_idx]:.3f}, init: {init_loss[net.best_guess_idx]:.3f}, final: {final_loss[net.best_guess_idx]:.3f}')
        if epoch % 1000 == 0:
            torch.save(net.state_dict(), save_dir + '/checkpoint_' + str(epoch) + '.pth')
            evaluate(net, wandb, save_dir, device)


    print('Finished Training')
    wandb.finish()
