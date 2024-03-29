"""
Visualization routines for progress reports

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from dataclasses import dataclass, field
from pathlib import Path
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../../')
from model import *

save_path = Path('../save')


# PLOT POINT CLOUDS
def plot_point_cloud(model, title='', save_path=None, add_noop=True, n_plots=4, max_bits=3, max_args=3, modulo=np.inf, return_pca=False, fig_width=4):
    width = fig_width
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * width, width))
    mpb = None

    # W = model.encoder_rnn.weight_hh_l0.data.numpy()
    # W = model.encoder_rnn.hh.weight.detach().numpy()
    # pca = PCA(n_components=2)
    # pca.fit(W)

    pca = None

    # TODO: plot along same PC's?
    if n_plots == 1:
        axs = np.array([axs])

    for n, ax in zip(range(10), axs.ravel()):
        params = [(i, j) for i in range(1, max_bits + 1) for j in range(1, max_args + 1)]
        if add_noop == False:
            n = 0

        ds = CurriculumDataset(params, max_noops=n, fix_noop=True)
        ds = CurriculumDatasetTrunc(ds, length=1000)

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
            lab_true = [out % modulo]
            lab_pred = [info['out']]
            all_labs_true.append(lab_true[0])
            all_labs_pred.append(lab_pred[0])

        all_points = np.concatenate(all_points, axis=-1)
        if pca == None:
            pca = PCA(n_components=2).fit(all_points.T)
            if return_pca:
                plt.clf()
                return pca

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
        plt.clf()

# <codecell>
@dataclass
class Case:
    name: str
    save_path: Path
    n_bits: int
    n_args: int
    max_noops: int = 0
    acc: int = 0
    mse: int = 0
    is_bin: bool = False


max_bits = 10
max_args = 10
n_batches = 4

all_cases = []
for n_bit in range(1, max_bits + 1):
    for n_arg in range(1, max_args + 1):
        all_cases.extend([
            Case(name='Bin 7bit', save_path='256d_7bit_bin', n_bits=n_bit, n_args=n_arg, max_noops=5, is_bin=True),
            Case(name='MSE 3bit', save_path='256d_3bit', n_bits=n_bit, n_args=n_arg, max_noops=5),  # TODO: fix noops?
            Case(name='MSE 7bit', save_path='256d_7bit', n_bits=n_bit, n_args=n_arg, max_noops=5),
            Case(name='MSE 10bit', save_path='256d_10bit', n_bits=n_bit, n_args=n_arg, max_noops=5),
            Case(name='Evo 3bit', save_path='evo/3bit', n_bits=n_bit, n_args=n_arg),
            Case(name='Evo (custom) 3bit', save_path='evo/3bit_custom', n_bits=n_bit, n_args=n_arg),
        ])

for case in tqdm(all_cases):
    if case.is_bin:
        model = RnnClassifierBinaryOut()
    else:
        model = RnnClassifier(0)

    model.load(save_path / case.save_path)

    ds = CurriculumDataset(params=[(case.n_bits, case.n_args)], max_noops=case.max_noops)
    dl = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=ds.pad_collate)
    dl_iter = iter(dl)
    for _ in range(n_batches):
        xs, ys = next(dl_iter)
        with torch.no_grad():
            if case.is_bin:
                preds = model.pretty_forward(xs)
                ys %= 2 ** model.n_places
            else:
                preds = model(xs)

        res = torch.isclose(preds.flatten(), ys, atol=0.5)

        case.acc += torch.mean(res.float()).item() / n_batches
        case.mse += torch.mean((preds.flatten() - ys)**2).item() / n_batches


df = pd.DataFrame(all_cases)

# <codecell>
plot_df = df[(df['name'] == 'Evo 3bit') \
      | (df['name'] == 'Evo (custom) 3bit') \
      | (df['name'] == 'MSE 3bit')]

# <codecell>
g = sns.catplot(data=plot_df, x='n_bits', y='acc', hue='name', col='n_args', col_wrap=3, kind='bar', height=2, aspect=1.5)
g.despine(left=True)
g.legend.set_title("")
plt.savefig('fig/evo_acc_comparison.png')

# %%
plt.clf()
g = sns.catplot(data=plot_df, x='n_bits', y='mse', hue='name', col='n_args', col_wrap=3, kind='bar', height=2, aspect=1.5)
g.despine(left=True)
g.set(yscale='log')
g.legend.set_title("")
plt.savefig('fig/evo_mse_comparison.png')

# <codecell>
plot_df = df[~df['name'].str.contains('^Evo')]
# %%
# <codecell>
g = sns.catplot(data=plot_df, x='n_bits', y='acc', hue='name', col='n_args', col_wrap=3, kind='bar', height=2, aspect=1.5)
g.despine(left=True)
g.legend.set_title("")
plt.savefig('fig/curr_acc_comparison.png')

# %%
plt.clf()
g = sns.catplot(data=plot_df, x='n_bits', y='mse', hue='name', col='n_args', col_wrap=3, kind='bar', height=2, aspect=1.5)
g.despine(left=True)
g.set(yscale='log')
g.legend.set_title("")
plt.savefig('fig/curr_mse_comparison.png')

# <codecell>
## FOR COSYNE: lengthwise generalization plot
curr_df = plot_df[plot_df['n_bits'] == 3]
g = sns.barplot(data=curr_df, x='n_args', y='acc', hue='name')
g.legend().set_title('')

# TODO: redo benchmarks with lengthwise and bitwise plots
# TODO: include (max) eigenvalues in benchmarking runs
# TODO: identify clean trajectory plot that shows stretching


# %% PRINT EIGENVALUES
with torch.no_grad():
    mod_1d = RnnClassifier(0).load('../save/1d_single')
    mod_3d = RnnClassifier(0).load('../save/3d_single')
    mod_256d = RnnClassifier(0).load('../save/256d_single')
    mod_10bit = RnnClassifier(0).load('../save/256d_10bit')

    val_1d = mod_1d.encoder_rnn.weight_hh_l0.item()
    val_3d = np.abs(np.max(np.linalg.eigvals(mod_3d.encoder_rnn.weight_hh_l0.numpy())))
    val_256d = np.abs(np.max(np.linalg.eigvals(mod_256d.encoder_rnn.weight_hh_l0.numpy())))
    val_10bit = np.abs(np.max(np.linalg.eigvals(mod_10bit.encoder_rnn.weight_hh_l0.numpy())))

    print('val_1d', val_1d)
    print('val_3d', val_3d)
    print('val_256d', val_256d)
    print('val_10bit', val_10bit)

    plt.bar([1,2,3,4], [val_1d, val_3d, val_256d, val_10bit])
    plt.gca().set_xticks([1,2,3,4])
    plt.gca().set_xticklabels(['RNN 1D', 'RNN 3D', 'RNN 256D', 'RNN 10bit'])
    plt.ylabel('Max eigenvalue')
    plt.savefig('fig/eig_comparison.png')
    plt.clf()

    xs = np.arange(20)
    vals = [torch.tensor([[1] + x * [0]]) for x in xs]

    pred_1d = [mod_1d(v).flatten().numpy() for v in vals]
    pred_3d = [mod_3d(v).flatten().numpy() for v in vals]
    pred_256d = [mod_256d(v).flatten().numpy() for v in vals]
    pred_10bit = [mod_10bit(v).flatten().numpy() for v in vals]
    true = [2 ** x for x in xs]

    alpha = 0.6
    plt.plot(xs, true, 'o--', color='black', label='True', alpha=alpha)
    plt.plot(xs, pred_1d, 'o--', label='RNN 1D', alpha=alpha)
    plt.plot(xs, pred_3d, 'o--', label='RNN 3D', alpha=alpha)
    plt.plot(xs, pred_256d, 'o--', label='RNN 256D', alpha=alpha)
    plt.plot(xs, pred_10bit, 'o--', label='RNN 10bit', alpha=alpha)

    plt.yscale('log')
    plt.legend()
    plt.xlabel('# zeros')
    plt.ylabel('Value')

    plt.savefig('fig/zeros_comparison.png')
    plt.clf()

    


# <codecell>
model = RnnClassifier(0).load('../save/256d_3bit')
plot_point_cloud(model, max_bits=3, save_path='fig/cloud_3bit_max3bit')
plot_point_cloud(model, max_bits=7, save_path='fig/cloud_3bit_max7bit')
plot_point_cloud(model, max_bits=10, save_path='fig/cloud_3bit_max10bit')

# <codecell>
model = RnnClassifier(0).load('../save/256d_7bit')
plot_point_cloud(model, max_bits=3, save_path='fig/cloud_7bit_max3bit')
plot_point_cloud(model, max_bits=7, save_path='fig/cloud_7bit_max7bit')
plot_point_cloud(model, max_bits=10, save_path='fig/cloud_7bit_max10bit')

# <codecell>
model = RnnClassifier(0).load('../save/256d_10bit')
plot_point_cloud(model, max_bits=3, save_path='fig/cloud_10bit_max3bit')
plot_point_cloud(model, max_bits=7, save_path='fig/cloud_10bit_max7bit')
plot_point_cloud(model, max_bits=10, save_path='fig/cloud_10bit_max10bit')

# <codecell>
model = RnnClassifier(0).load('../save/evo/3bit')
plot_point_cloud(model, max_bits=3, add_noop=False, save_path='fig/cloud_evo_max3bit', n_plots=1)
plot_point_cloud(model, max_bits=7, add_noop=False, save_path='fig/cloud_evo_max7bit', n_plots=1)
plot_point_cloud(model, max_bits=10,add_noop=False,  save_path='fig/cloud_evo_max10bit', n_plots=1)

# <codecell>
model = RnnClassifier(0).load('../save/256d_single')
plot_point_cloud(model, max_bits=3, save_path='fig/cloud_single_max3bit', n_plots=1, add_noop=False, max_args=1)
plot_point_cloud(model, max_bits=7, save_path='fig/cloud_single_max7bit', n_plots=1, add_noop=False, max_args=1)
plot_point_cloud(model, max_bits=10, save_path='fig/cloud_single_max10bit', n_plots=1, add_noop=False, max_args=1)

# <codecell>
model = RnnClassifierBinaryOut().load('../save/256d_7bit_bin')
plot_point_cloud(model, max_bits=3, save_path='fig/cloud_7bit_bin_max3bit', modulo=16)
plot_point_cloud(model, max_bits=7, save_path='fig/cloud_7bit_bin_max7bit', modulo=16)
plot_point_cloud(model, max_bits=10, save_path='fig/cloud_7bit_bin_max10bit', modulo=16)

# <codecell>
## FOR COSYNE
model = RnnClassifierBinaryOut().load('../save/256d_7bit_bin')
plt.rc('font', size=21)
plot_point_cloud(model, max_bits=7, save_path='fig/cloud_7bit_bin_max7bit.svg', modulo=16, fig_width=3)

# <codecell>
# PLOT TRAJECTORIES
# TODO: adapt
def tanh_deriv(x):
    return 1 - torch.tanh(x) ** 2

def relu_deriv(x):
    return (x > 0).float()


@torch.no_grad()
def compute_jacob(model, h, tok_idx, activ=torch.relu, activ_d=relu_deriv):
    h = torch.tensor(h).reshape(-1, 1).float()
    emb = model.get_embedding([tok_idx])
    W = model.encoder_rnn.weight_hh_l0
    F = -h + activ(W @ h + emb)

    deriv = activ_d(W @ h + emb).tile(1, W.shape[1])
    J = -torch.eye(W.shape[1]) + deriv * W
    return F, J

@torch.no_grad()
def compute_grads(model, h, tok_idx, **kwargs):
    F, J = compute_jacob(model, h, tok_idx, **kwargs)

    q = 0.5 * torch.norm(F) ** 2
    grad = J @ F
    hess = J @ J.T
    return q, grad, hess

@torch.no_grad()
def make_funcs(model, tok_idx, **kwargs):
    def q(x): return compute_grads(model, x, tok_idx, **kwargs)[0].numpy()
    def jac(x): return compute_grads(model, x, tok_idx, **kwargs)[1].flatten().numpy()
    def hess(x): return compute_grads(model, x, tok_idx, **kwargs)[2].numpy()

    return q, jac, hess

@torch.no_grad()
def optim(model, seq, h_init_idx, tok_idx, **kwargs):
    info = model.trace(seq)
    q, jac, hess = make_funcs(model, tok_idx, **kwargs)
    h_start = info['enc']['hidden'][h_init_idx].detach().numpy()
    res = minimize(q, h_start, method='trust-ncg', jac=jac, hess=hess)

    if not res['success']:
        res = minimize(q, h_start, method='bfgs', jac=jac)

    return res

def find_crit_points(model, all_seqs, **kwargs):
    all_results = []
    for seq in all_seqs:
        curr_results = []
        print('Compute seq:', seq)
        for i in tqdm(range(len(seq))):
            res = optim(model, seq, i, 5, **kwargs)
            curr_results.append(res)
        all_results.append(curr_results)
    
    return all_results

def plot_seq(model, all_seqs, all_results=None, save_path=None, pca=None, is_bin=False):
    all_trajs = []
    all_labs = []
    if all_results == None:
        all_results = [[]] * len(all_seqs)

    W = model.encoder_rnn.weight_hh_l0.data.numpy()
    if pca == None:
        pca = PCA(n_components=2)
        pca.fit(np.linalg.matrix_power(W, 1))

    for seq in all_seqs:
        seq = torch.tensor(seq)
        with torch.no_grad():
            info = model.trace(seq)

        traj = torch.cat(info['enc']['hidden'], axis=-1)
        # labs = model.readout(traj.T).argmax(axis=-1)
        if is_bin:
            raw = model.readout(traj.T)
            labs = model.raw_to_dec(raw)
        else:
            labs = model.readout(traj.T).flatten().round(decimals=1)
        all_labs.append([0] + labs.tolist())

        traj = traj.numpy()
        traj = pca.transform(traj.T).T
        all_trajs.append(traj)

    plt.gcf().set_size_inches(6, 6)
    lss = ['-', ':'] * len(all_seqs)
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
            plt.annotate(f'{lab:.1f}', point + 0.1, color='red')

        plt.annotate(f'{labs[-1]:.1f}', traj.T[-1] + 0.1, color='red')
        
        # plt.plot(traj[0,:] + jit_x, traj[1,:] + jit_y, 'o-', label=str(seq), alpha=0.8)


        for i, res in enumerate(result):
            if res['success']:
                crit_pt = pca.transform(res['x'].reshape(1, -1))
                plt.plot(crit_pt[0,0], crit_pt[0,1], 'ro', markersize=7, alpha=0.6)

                label = model.readout(torch.tensor(res['x'].reshape(1, -1)).float()).argmax(dim=-1)
                if label.item() <= 8:
                    offset = 0.15
                else:
                    offset = np.array([-0.45, 0.35]).reshape(2, 1)

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
        plt.clf()


# <codecell>
### 3 bit
seq = [1,5,5,5] + 10 * [2, 1, 5, 5, 5]
seq = [seq]

model = RnnClassifier(0).load('../save/256d_3bit')
pca = plot_point_cloud(model, max_bits=3, return_pca=True)
# pts = find_crit_points(model, seq)
plot_seq(model, all_seqs=seq, all_results=None, pca=pca, save_path='fig/traj_3bit_ones.png')
# <codecell>
seq = [1, 0, 5, 5, 5] + 5 * [2, 1, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_3bit_twos.png')
# <codecell>
seq = [1, 0, 5, 5, 5] + 5 * [2, 1, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_3bit_twos.png')

# <codecell>
seq = [1, 0, 0, 5, 5, 5] + 5 * [2, 1, 0, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_3bit_threes.png')

# <codecell>
seq = [
        # [1, 5, 5, 5, 2, 1, 0, 5, 5, 5, 2, 1, 0, 5, 5, 5, 2, 1, 5, 5, 5],
        # [1, 5, 5, 5, 2, 1, 0, 0, 0, 5, 5, 5, 2, 1]
        # [1, 5, 5, 5] + 15 * [1, 5, 5, 5],
        [1, 0, 0, 0, 0, 5, 5, 5]
]
plot_seq(model, pca=pca, all_seqs=seq, save_path='fig/traj_3bit_attempt5bit.png')

# <codecell>
### 7 bit
seq = [1,5,5,5] + 10 * [2, 1, 5, 5, 5]
seq = [seq]

model = RnnClassifier(0).load('../save/256d_7bit')
pca = plot_point_cloud(model, max_bits=3, return_pca=True)
# pts = find_crit_points(model, seq)
plot_seq(model, all_seqs=seq, all_results=None, pca=pca, save_path='fig/traj_7bit_ones.png')
# <codecell>
seq = [1, 0, 5, 5, 5] + 5 * [2, 1, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_7bit_twos.png')

# <codecell>
seq = [1, 0, 0, 5, 5, 5] + 5 * [2, 1, 0, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_7bit_threes.png')

# <codecell>
seq = [
        # [1, 5, 5, 5, 2, 1, 0, 5, 5, 5, 2, 1, 0, 5, 5, 5, 2, 1, 5, 5, 5],
        # [1, 5, 5, 5, 2, 1, 0, 0, 0, 5, 5, 5, 2, 1]
        # [1, 5, 5, 5] + 15 * [1, 5, 5, 5],
        [1, 0, 0, 0, 0, 5, 5, 5],
        [1, 0, 0, 0, 0, 0, 0, 5, 5, 5],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5]
]
plot_seq(model, pca=pca, all_seqs=seq, save_path='fig/traj_7bit_16_64_256_attempt.png')

# <codecell>
### 10 bit
seq = [1,5,5,5] + 10 * [2, 1, 5, 5, 5]
seq = [seq]

model = RnnClassifier(0).load('../save/256d_10bit')
pca = plot_point_cloud(model, max_bits=3, return_pca=True)
# pts = find_crit_points(model, seq)
plot_seq(model, all_seqs=seq, all_results=None, pca=pca, save_path='fig/traj_10bit_ones.png')
# <codecell>
seq = [1, 0, 5, 5, 5] + 5 * [2, 1, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_10bit_twos.png')
# <codecell>
seq = [1, 0, 5, 5, 5] + 5 * [2, 1, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_10bit_twos.png')

# <codecell>
seq = [1, 0, 0, 5, 5, 5] + 5 * [2, 1, 0, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_10bit_threes.png')

# <codecell>
seq = [
        # [1, 5, 5, 5, 2, 1, 0, 5, 5, 5, 2, 1, 0, 5, 5, 5, 2, 1, 5, 5, 5],
        # [1, 5, 5, 5, 2, 1, 0, 0, 0, 5, 5, 5, 2, 1]
        # [1, 5, 5, 5] + 15 * [1, 5, 5, 5],
        [1, 0, 0, 0, 0, 5, 5, 5],
        [1, 0, 0, 0, 0, 0, 0, 5, 5, 5],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5]
]
plot_seq(model, pca=pca, all_seqs=seq, save_path='fig/traj_10bit_16_64_256_1024_attempt.png')


# <codecell>
seq = [
        # [1, 5, 5, 5, 2, 1, 0, 5, 5, 5, 2, 1, 0, 5, 5, 5, 2, 1, 5, 5, 5],
        # [1, 5, 5, 5, 2, 1, 0, 0, 0, 5, 5, 5, 2, 1]
        # [1, 5, 5, 5] + 15 * [1, 5, 5, 5],
        [1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0]
]
model = RnnClassifier(0).load('../save/256d_single')
pca = plot_point_cloud(model, max_bits=3, return_pca=True)
# pts = find_crit_points(model, seq)
plot_seq(model, all_seqs=seq, all_results=None, pca=pca, save_path='fig/traj_single.png')


# %%
### BINARY
seq = [1, 5, 5, 5] + 2 * [2, 1, 5, 5, 5]

model = RnnClassifierBinaryOut().load('../save/256d_7bit_bin')
pca = plot_point_cloud(model, max_bits=3, return_pca=True)
plot_seq(model, all_seqs=[seq], all_results=None, pca=pca, is_bin=True, save_path='fig/traj_bin_ones.png')

# <codecell>
seq = [1, 0, 5, 5, 5] + 2 * [2, 1, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_bin_twos.png', is_bin=True)

# <codecell>
seq = [1, 0, 0, 5, 5, 5] + 2 * [2, 1, 0, 0, 5, 5, 5]
plot_seq(model, pca=pca, all_seqs=[seq], save_path='fig/traj_bin_threes.png', is_bin=True)

# <codecell>
seq = [
    [1, 0, 5, 5, 5],
    [1, 0, 0, 5, 5, 5],
    [1, 0, 0, 0, 5, 5, 5],
    [1, 0, 0, 0, 0, 5, 5, 5],
    [1, 0, 0, 0, 0, 5, 5, 5, 2, 1],
    [1, 0, 0, 0, 0, 5, 5, 5, 2, 1, 0, 5, 5, 5],
]
plot_seq(model, pca=pca, all_seqs=seq, is_bin=True, save_path='fig/traj_bin_various.png')