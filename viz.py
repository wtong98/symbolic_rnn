"""
Inspect the operation of the model

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from model import *

# %%
ds = BinaryAdditionDataset()

model = BinaryAdditionLSTM()
model.load('save/mini')

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

seq = [3, 1, 1, 0, 2, 1, 0, 3]

# <codecell>
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
plt.savefig('save/fig/mini_state.png')
print(info['out'])

# <codecell>
### PLOT READOUT
fig, axs = plt.subplots(1, 5, figsize=(15, 3))

for i, ax in enumerate(axs.ravel()):
    ax.bar(np.arange(model.hidden_size), model.readout.weight.data[:,i])
    ax.set_xticks(np.arange(model.hidden_size))
    ax.set_xticklabels(ds.idx_to_token)
    ax.set_title(f'Cell: {i}')

# TODO: overlay contribution to readout over cell above ^
fig.tight_layout()
plt.savefig('save/fig/mini_readout.png')

# <codecell>
### PLOT RELATION TO INPUT SEQ
seq = [3, 1, 1, 0, 2, 1, 0, 3]
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
    plot_states(states, seq, ds, concat=False, ax=ax)



fig.tight_layout()
plt.savefig('save/fig/mini_saliency.png')

# %%
