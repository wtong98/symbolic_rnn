"""
Generate plots for COSYNE
"""

# <codecell>
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm

import sys
sys.path.append('../')
from model import *

# <codecell>
### Point clouds
def plot_point_cloud(model, title='', save_path=None):
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    # fig, axs = plt.subplots(4, 1, figsize=(2, 8))
    mpb = None

    W = model.encoder_rnn.weight_hh_l0.data.numpy()
    # W = model.encoder_rnn.hh.weight.detach().numpy()
    pca = PCA(n_components=2)
    pca.fit(W)

    # TODO: plot along same PC's?
    for n, ax in zip(range(10), axs.ravel()):
        # n *= 20
        ds = BinaryAdditionDataset(n_bits=3, 
                                onehot_out=True, 
                                use_zero_pad=True,
                                max_args=3, 
                                add_noop=True,
                                max_noop=n,
                                max_noop_only=True,
                                little_endian=False)

        all_points = []
        all_labs_true = []
        all_labs_pred = []

        for seq, out in ds:
            seq = torch.tensor(seq)
            with torch.no_grad():
                info = model.trace(seq)
            
            point = info['enc']['hidden'][-1].numpy()
            all_points.append(point)

            # lab_true = ds.tokens_to_args(out)
            # lab_pred = ds.tokens_to_args(info['out'])
            lab_true = [out]
            lab_pred = [info['out']]
            all_labs_true.append(lab_true[0])
            all_labs_pred.append(lab_pred[0])

        all_points = np.concatenate(all_points, axis=-1)
        # all_points = PCA(n_components=2).fit_transform(all_points.T).T
        all_points = pca.transform(all_points.T).T

        mpb = ax.scatter(all_points[0,:], all_points[1,:], c=all_labs_true)
        ax.set_title(f'noops = {n}')
        ax.axis('off')
        ax.grid(True)
        # ax.patch.set_edgecolor('gray')
        # ax.patch.set_linewidth('0.5')

    fig.colorbar(mpb)
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)

# <codecell>
model = RnnClassifier(0)
model.load('../save/hid100k_vargs3_nbits3')
plot_point_cloud(model, save_path='cosyne_fig/cloud_full.svg', title='tanh + classification')

# <codecell>
model = RnnClassifier(0)
model.load('../save/skip_6')
plot_point_cloud(model, save_path='cosyne_fig/cloud_skip_6.svg', title='tanh + classification, skip 6 ("110")')

# <codecell>
model = RnnClassifier(0, loss_func='mse', nonlinearity='tanh')
model.load('../save/relu_mse_interleaved')
plot_point_cloud(model, save_path='cosyne_fig/cloud_relu_mse.svg', title='relu + mse')

# <codecell>
# NOTE: zero's extrapolation relegated to `training_rig.py`
# NOTE: length extrapolation relegated to `benchmark.py`
# TRAJECTORY PLOT

def tanh_deriv(x):
    return 1 - torch.tanh(x) ** 2

@torch.no_grad()
def compute_jacob(model, h, tok_idx):
    h = torch.tensor(h).reshape(-1, 1).float()
    emb = model.get_embedding([tok_idx])
    W = model.encoder_rnn.weight_hh_l0
    F = -h + torch.tanh(W @ h + emb)

    deriv = tanh_deriv(W @ h + emb).tile(1, W.shape[1])
    J = -torch.eye(W.shape[1]) + deriv * W
    return F, J

@torch.no_grad()
def compute_grads(model, h, tok_idx):
    F, J = compute_jacob(model, h, tok_idx)

    q = 0.5 * torch.norm(F) ** 2
    grad = J @ F
    hess = J @ J.T
    return q, grad, hess

@torch.no_grad()
def make_funcs(model, tok_idx):
    def q(x): return compute_grads(model, x, tok_idx)[0].numpy()
    def jac(x): return compute_grads(model, x, tok_idx)[1].flatten().numpy()
    def hess(x): return compute_grads(model, x, tok_idx)[2].numpy()

    return q, jac, hess

@torch.no_grad()
def optim(model, seq, h_init_idx, tok_idx):
    info = model.trace(seq)
    q, jac, hess = make_funcs(model, tok_idx)
    h_start = info['enc']['hidden'][h_init_idx].detach().numpy()
    res = minimize(q, h_start, method='trust-ncg', jac=jac, hess=hess)

    if not res['success']:
        res = minimize(q, h_start, method='bfgs', jac=jac)

    return res

def find_crit_points(model, all_seqs):
    all_results = []
    for seq in all_seqs:
        curr_results = []
        print('Compute seq:', seq)
        for i in tqdm(range(len(seq))):
            res = optim(model, seq, i, 5)
            curr_results.append(res)
        all_results.append(curr_results)
    
    return all_results

def plot_seq(model, all_seqs, all_results=None, save_path=None):
    all_trajs = []
    all_labs = []
    if all_results == None:
        all_results = [[]]  # TODO: temporary

    W = model.encoder_rnn.weight_hh_l0.data.numpy()
    pca = PCA(n_components=2)
    pca.fit(np.linalg.matrix_power(W, 1))

    for seq in all_seqs:
        seq = torch.tensor(seq)
        with torch.no_grad():
            info = model.trace(seq)

        traj = torch.cat(info['enc']['hidden'], axis=-1)
        labs = model.readout(traj.T).argmax(axis=-1)
        all_labs.append(labs.tolist())

        traj = traj.numpy()
        traj = pca.transform(traj.T).T
        all_trajs.append(traj)

    plt.gcf().set_size_inches(6, 6)
    lss = ['-', ':']
    for seq, traj, labs, result, ls in zip(all_seqs, all_trajs, all_labs, all_results, lss):
        # traj = pca.transform(traj.T).T
        traj = np.concatenate((np.zeros((2, 1)), traj), axis=-1)
        # jit_x = np.random.uniform(-1, 1) * 0.05
        # jit_y = np.random.uniform(-1, 1) * 0.05
        jit_x = 0
        jit_y = 0

        diffs = traj[:,1:] - traj[:,:-1]
        for i, point, lab, diff in zip(range(len(seq)), traj.T, labs, diffs.T):
            if seq[i] == 5:
                color = 'C0'
            elif seq[i] == 1:
                color = 'C1'
            elif seq[i] == 0:
                color = 'C4'
            else:
                color = 'C2'
            
            point[0] += jit_x
            point[1] += jit_y
            plt.arrow(*point, *diff, color=color, alpha=0.8, length_includes_head=True, width=0.03, head_width=0.25, head_length=0.2)
            # plt.annotate(lab, point + 0.1)
        
        # plt.plot(traj[0,:] + jit_x, traj[1,:] + jit_y, 'o-', label=str(seq), alpha=0.8)


        for i, res in enumerate(result):
            if res['success']:
                crit_pt = pca.transform(res['x'].reshape(1, -1))
                plt.plot(crit_pt[0,0], crit_pt[0,1], 'ro', markersize=7, alpha=0.6)

                label = model.readout(torch.tensor(res['x'].reshape(1, -1)).float()).argmax(dim=-1)
                if label.item() <= 8:
                    offset = 0.15
                else:
                    offset = np.array([-0.35, 0.25]).reshape(2, 1)

                plt.annotate(label.item(), crit_pt.T + offset, color='red')

                # orig_pt = traj[:,i]
                # plt.arrow(orig_pt[0], orig_pt[1], crit_pt[0,0] - orig_pt[0], crit_pt[0,1] - orig_pt[1], alpha=0.2)

    plt.annotate('Start', (0.12, -0.1), color='red')
    plt.axis('off')


    handles = [
        mpatches.Patch(color='C0', label='No-Op'),
        mpatches.Patch(color='C1', label='1'),
        mpatches.Patch(color='C4', label='0'),
        mpatches.Patch(color='C2', label='+'),
        mpatches.Patch(color='red', label='Crit. Pt')
    ]

    # plt.legend(handles=handles)
    if save_path:
        plt.savefig(save_path)


# <codecell>
model = RnnClassifier(0)
model.load('save/hid100k_vargs3_nbits3')
all_seqs = [[1,5,5,5] + 15 * [2,1,5,5,5]]
all_results = find_crit_points(model, all_seqs)
plot_seq(model, all_seqs, all_results, save_path='viz/cosyne_fig/traj_full.svg')

# <codecell>
model = RnnClassifier(0)
model.load('save/skip_6')
all_seqs = [[1,1,0,5,5,5,2,1,5,5,5]]
all_results = find_crit_points(model, all_seqs)
plot_seq(model, all_seqs, all_results, save_path='viz/cosyne_fig/traj_skip_6.svg')

# <codecell>
### 1D RNN loss landscape