"""
Inspect the operation of the model

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model import *

# %%
ds = BinaryAdditionDataset(n_bits=2)

model = BinaryAdditionLSTM()
model.load('save/medium_2k')

# <codecell>
### PLOT EMBEDDINGS # TODO: make it cleaner
weights = model.embedding.weight.data.numpy()
plt.scatter(weights[:,0], weights[:,1], c=['b', 'r', 'y', 'g', 'k'])


### PLOT STATES
def plot_states(states, seq, ds, ax=None, concat=True):
    if ax == None:
        ax = plt.gca()

    if concat:
        states = torch.concat(states, dim=-1)
        
    ax.imshow(states)

    symbols = [ds.idx_to_token[s] for s in seq]
    ax.set_xticks(np.arange(states.shape[1]))
    ax.set_xticklabels(symbols[:states.shape[1]])

    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            ax.text(j, i, f'{states[i, j].item():.2f}',
                        ha="center", va="center", color="w")

# <codecell>
seq = [3, 1, 0, 2, 1, 0, 3]

with torch.no_grad():
    info = model.trace(seq)

fig, axs = plt.subplots(2, 6, figsize=(22, 8))
names = [
    ('f', 'forget gate'),
    ('i', 'input gate'),
    ('g', 'write content'),
    ('o', 'output gate'),
    ('cell', 'cell state'),
    ('hidden', 'hidden state')
]

parts = [
    ('enc', 'encoder'),
    ('dec', 'decoder')
]

for (p_idx, p_name), p_axs in zip(parts, axs):
    for (n_idx, n_name), ax in zip(names, p_axs):
        sequence = seq if p_idx == 'enc' else info['out']
        plot_states(info[p_idx][n_idx], sequence, ds, ax=ax)
        ax.set_title(n_name)

fig.tight_layout()
# plt.savefig('save/fig/micro_state_128k.png')
print(info['out'])

# <codecell>
### PLOT READOUT
fig, axs = plt.subplots(1, model.hidden_size, figsize=(3*model.hidden_size, 3))

for i, ax in enumerate(axs.ravel()):
    ax.bar(np.arange(5), model.readout.weight.data[:,i])
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(ds.idx_to_token)
    ax.set_title(f'Cell: {i}')

# TODO: overlay contribution to readout over cell above ^
fig.tight_layout()
plt.savefig('save/fig/micro_readout.png')

# <codecell>
### PLOT RELATION TO INPUT SEQ
cell_idxs = np.arange(model.hidden_size)

enc_sal_maps = defaultdict(list)
for i in range(len(seq)):
    for c in cell_idxs:
        info = model.trace(seq)
        info['input_emb'].retain_grad()
        deriv_map = torch.nn.functional.one_hot(torch.tensor([c]), num_classes=model.hidden_size).T
        info['enc']['cell'][i].backward(deriv_map)

        total_grad = torch.torch.linalg.norm(info['input_emb'].grad, dim=-1)
        enc_sal_maps[c].append(total_grad)

len_out = len(info['out'])
print('out', info['out'])
dec_sal_maps = defaultdict(list)

for i in range(len_out - 1):
    for c in cell_idxs:
        info = model.trace(seq)
        for emb in info['output_emb']:
            emb.retain_grad()

        deriv_map = torch.nn.functional.one_hot(torch.tensor([c]), num_classes=model.hidden_size).T
        info['dec']['cell'][i].backward(deriv_map)

        total_grad = [torch.linalg.norm(emb.grad, dim=-1) if emb.grad != None else 0 for emb in info['output_emb']]
        total_grad = torch.tensor(total_grad)
        dec_sal_maps[c].append(total_grad)

print('out', info['out'])

# <codecell>
fig, axs = plt.subplots(2, len(seq), figsize=(4 * len(seq), 8))

for i, ax in enumerate(axs[0].ravel()):
    states = [enc_sal_maps[c][i] for c in range(model.hidden_size)]
    states = torch.stack(states, dim=0)
    plot_states(states, seq, ds, concat=False, ax=ax)
    ax.set_title(f'timestep: {i}')

for i, ax in enumerate(axs[1][:len(info['output_emb'])]):
    states = [dec_sal_maps[c][i] for c in range(model.hidden_size)]
    states = torch.stack(states, dim=0)
    plot_states(states, info['out'], ds, concat=False, ax=ax)



fig.tight_layout()
plt.savefig('save/fig/micro_saliency.png')

# %%
### PLOT TRAJECTORIES THROUGH CELL SPACE
all_seqs = []
all_trajs = []

test_seqs = [
    # 0 block
    # [3, 0, 2, 0, 3],
    # [3, 0, 0, 2, 0, 3],
    # [3, 0, 2, 0, 0, 3],
    # [3, 0, 0, 2, 0, 0, 3],

    # 1 block
    # [3, 0, 2, 1, 3],
    # [3, 0, 2, 0, 1, 3],
    # [3, 0, 0, 2, 1, 3],
    # [3, 0, 0, 2, 0, 1, 3],

    # 1 block
    # [3, 1, 2, 0, 3],
    # [3, 0, 1, 2, 0, 3],
    # [3, 1, 2, 0, 0, 3],
    # [3, 0, 1, 2, 0, 0, 3],

    # 2 block
    [3, 1, 2, 1, 3],
    [3, 0, 1, 2, 1, 3],
    [3, 1, 2, 0, 1, 3],
    [3, 0, 1, 2, 0, 1, 3],
    [3, 0, 0, 2, 1, 1, 3],
    [3, 1, 1, 2, 0, 0, 3],

    # 3 block
    # [3, 1, 1, 2, 1, 3],
    # [3, 1, 1, 2, 0, 1, 3],
    # [3, 1, 2, 1, 1, 3],
    # [3, 0, 1, 2, 1, 1, 3],
]

for seq in test_seqs:
    seq = torch.tensor(seq)
    with torch.no_grad():
        info = model.trace(seq)
    
    traj = torch.cat(info['enc']['cell'], axis=1).numpy()
    all_trajs.append(traj)
    all_seqs.append(seq.numpy())

trajs_blob = np.concatenate(all_trajs.copy(), axis=-1)
pca = PCA(n_components=2)
pca.fit_transform(trajs_blob.T)

plt.gcf().set_size_inches(12, 12)
for seq, traj in zip(all_seqs, all_trajs):
    traj = pca.transform(traj.T).T
    jit_x = np.random.uniform() * 0.04
    jit_y = np.random.uniform() * 0.04
    plt.plot(traj[0,:] + jit_x, traj[1,:] + jit_y, 'o-', label=str(seq), alpha=0.8)

plt.legend()
# plt.savefig('save/fig/micro_128k_traj_2.png')

# %%
### PLOT CLOUD OF FINAL CELL STATES BY VALUE
all_points = []
all_labs_true = []
all_labs_pred = []

for seq, out in ds:
    seq = torch.tensor(seq)
    with torch.no_grad():
        info = model.trace(seq)
    
    point = info['enc']['cell'][-1].numpy()
    all_points.append(point)

    lab_true = ds.tokens_to_args(out)
    lab_pred = ds.tokens_to_args(info['out'])
    all_labs_true.append(lab_true[0])
    all_labs_pred.append(lab_pred[0])

all_points = np.concatenate(all_points, axis=-1)
all_points = PCA(n_components=2).fit_transform(all_points.T).T

plt.scatter(all_points[0,:], all_points[1,:], c=all_labs_true)
plt.legend()
# plt.savefig('save/fig/micro_128k_cell_cloud.png')

# <codecell>
plt.scatter(all_points[0,:], all_points[1,:], c=all_labs_pred)
plt.legend()

# %%
