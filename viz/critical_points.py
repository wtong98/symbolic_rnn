"""
Find the critical / slow points of the system, and dynamics between them
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
ds = BinaryAdditionDataset(n_bits=3, 
                           onehot_out=True, 
                           max_args=3, 
                           add_noop=True,
                           max_noop=5,
                           little_endian=False)

model = RnnClassifier(max_arg=6)
model.load('../save/hid100k_vargs3_nbits3')


# <codecell>
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

# <codecell>
all_seqs = [
    # [1, 5, 5, 5] + 20 * [2, 1, 5, 5, 5],
    # [1, 0, 5, 5, 5] + 9 * [2, 1, 0, 5, 5, 5]
    [1, 0, 0, 0, 5, 5, 5] + [2, 1, 5, 5, 5],
    [1, 0, 0, 5, 5, 5] + [2, 1, 0, 0, 5, 5, 5] + [2, 1, 5, 5, 5]
]

all_results = []
for seq in all_seqs:
    curr_results = []
    print('Compute seq:', seq)
    for i in tqdm(range(len(seq))):
        res = optim(model, seq, i, 5)
        curr_results.append(res)
    all_results.append(curr_results)

# <codecell>
all_trajs = []

W = model.encoder_rnn.weight_hh_l0.data.numpy()
pca = PCA(n_components=2)
pca.fit(np.linalg.matrix_power(W, 1))

for seq in all_seqs:
    seq = torch.tensor(seq)
    with torch.no_grad():
        info = model.trace(seq)

    traj = torch.cat(info['enc']['hidden'], axis=1).numpy()
    traj = pca.transform(traj.T).T
    all_trajs.append(traj)

plt.gcf().set_size_inches(12, 12)
lss = ['-', ':']
for seq, traj, result, ls in zip(all_seqs, all_trajs, all_results, lss):
    # traj = pca.transform(traj.T).T
    traj = np.concatenate((np.zeros((2, 1)), traj), axis=-1)
    jit_x = np.random.uniform(-1, 1) * 0.05
    jit_y = np.random.uniform(-1, 1) * 0.05
    # jit_x = 1
    # jit_y = 1

    diffs = traj[:,1:] - traj[:,:-1]
    for i, point, diff in zip(range(len(seq)), traj.T, diffs.T):
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
        plt.arrow(*point, *diff, color=color, linestyle=ls, linewidth=2, alpha=0.85)
    
    # plt.plot(traj[0,:] + jit_x, traj[1,:] + jit_y, 'o-', label=str(seq), alpha=0.8)


    for i, res in enumerate(result):
        if res['success']:
            crit_pt = pca.transform(res['x'].reshape(1, -1))
            plt.plot(crit_pt[0,0], crit_pt[0,1], 'ro', markersize=10, alpha=0.2)

            label = model.readout(torch.tensor(res['x'].reshape(1, -1)).float()).argmax(dim=-1)
            plt.annotate(label.item(), crit_pt.T + 0.1)

            orig_pt = traj[:,i]
            plt.arrow(orig_pt[0], orig_pt[1], crit_pt[0,0] - orig_pt[0], crit_pt[0,1] - orig_pt[1], alpha=0.2)

plt.annotate('enc start', (0, 0))


handles = [
    mpatches.Patch(color='C0', label='No-Op'),
    mpatches.Patch(color='C1', label='1'),
    mpatches.Patch(color='C4', label='0'),
    mpatches.Patch(color='C2', label='+'),
    mpatches.Patch(color='gray', label='To min'),
    mpatches.Patch(color='red', label='Crit. Pt')
]

plt.legend(handles=handles)
plt.savefig('../save/fig/rnn_noop_cloud_with_crit_full_comp.png')
# %%
