"""
Wide-ranging exploration of model weights, for interpretability
"""

# <codecell>
from asyncore import read
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import ConcatDataset

import sys
sys.path.append('../')

from model import *

# <codecell>

ds_full = BinaryAdditionDataset(n_bits=3, 
                           onehot_out=True, 
                           max_args=3, 
                           add_noop=True,
                           max_noop=5,
                           use_zero_pad=True,
                           float_labels=True,
                        #    max_noop_only=True,
                        #    max_only=True, 
                           little_endian=False,
                           filter_={
                               'in_args': []
                           })

ds_args_only = BinaryAdditionDataset(n_bits=7, 
                           onehot_out=True, 
                           max_args=1, 
                           add_noop=True,
                           max_noop=5,
                           use_zero_pad=True,
                           float_labels=True,
                        #    max_noop_only=True,
                        #    max_only=True, 
                           little_endian=False,
                           filter_={
                               'in_args': []
                           })

def make_dl(ds):
    train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    return train_dl, test_dl

ds_all = ConcatDataset([ds_args_only, ds_full])
ds_all.pad_collate = ds_args_only.pad_collate
train_dl, test_dl = make_dl(ds_all)

model = RnnClassifier(0)
model.load('../save/relu_mse_interleaved_linear_readout')

# <codecell>
### INVESTIGATE DIMENSIONALITY OF NEURAL ACTIVATIONS
with torch.no_grad():
    all_activ = torch.concat([model.encode(x) for x, _ in train_dl], axis=0)
    all_activ = all_activ.numpy()

all_activ = StandardScaler().fit_transform(all_activ)
pca = PCA().fit(all_activ)
plt.plot(np.arange(1, 257), np.cumsum(pca.explained_variance_ratio_), 'o--')
plt.xlim((1, 30))
plt.xticks(np.arange(1, 30)[0::3])

plt.title('Neural activations are relatively low dimensional')
plt.ylabel('Proportion of explained variance')
plt.xlabel('PCs')

# plt.savefig('../save/fig/prop_exp_var.png')

# <codecell>
### Which directions do these PCs point?
pc_inv = np.zeros((model.hidden_size, model.hidden_size))
np.fill_diagonal(pc_inv, 1)
pc_inv = pca.inverse_transform(pc_inv)

sort_idxs = np.argsort(pc_inv[3,:])
plt.bar(np.arange(model.hidden_size), pc_inv[3,:][sort_idxs])
# plt.hist(pc_inv[0,:])

# <codecell>
### How much does each dimension contribute to the final value?
w_readout = model.readout.weight.data.numpy()
# b_readout = model.readout.bias.data.numpy()
# w_readout @ pc_inv[0,:].reshape(-1, 1)

plt.bar(np.arange(256), w_readout.flatten())

# <codecell>
### Dimensionality of readout operations
# (Apparently it's quite high)
W = model.hidden.weight.data.numpy()[:256]
w0 = W[8,:].reshape(-1, 1)

norms = np.linalg.norm(W, axis=1).reshape(-1, 1)
sims = W @ w0 / norms / np.linalg.norm(w0)

plt.bar(np.arange(256), np.sort(sims.flatten()))


# <codecell>
### INVESTIGATE RECURRENT CONNECTIONS
token_ids = torch.tensor([0, 1, 2, 5])

with torch.no_grad():
    emb = model.embedding(token_ids).numpy().T

    W = model.encoder_rnn.weight_hh_l0.data.numpy()
    W_in = model.encoder_rnn.weight_ih_l0.data.numpy()
    b = model.encoder_rnn.bias_ih_l0.data.numpy() + model.encoder_rnn.bias_hh_l0.data.numpy()

    emb = W_in @ emb + b.reshape(-1, 1)

gamma = 0.9
W_sum = np.sum([(gamma * W) ** k for k in range(1, 10)], axis=0)

emb_0_idx = emb[:,0].argsort()
emb_1_idx = emb[:,1].argsort()
emb_2_idx = emb[:,2].argsort()
emb_5_idx = emb[:,3].argsort()

bounds = 0.1
plt.imshow(W[:,emb_1_idx][emb_1_idx,:], cmap='bwr', vmin=-bounds, vmax=bounds)
# plt.imshow(W[:,emb_1_idx], cmap='Spectral')
plt.colorbar()
plt.savefig('../save/fig/w_1.png')


def plot_seq(seq, emb_idx, bound=0.1):
    n_toks = len(seq)
    fig, ax = plt.subplots(1, n_toks, figsize=(n_toks * 3, 3))

    h = np.zeros((256, 1))
    for i, tok in enumerate(seq):
        h = np.maximum(W @ h + emb[:,tok].reshape(-1, 1), 0)
        W_prod = W * np.tile(h.reshape(1, -1), (256, 1))
        mpb = ax[i].imshow(W_prod[:,emb_idx][emb_idx,:], cmap='bwr', vmin=-bound, vmax=bound)
        ax[i].set_title(f'Tok: {tok}')
        # fig.colorbar(mpb, ax=ax[i])
    
    answ = model.readout(torch.tensor(h.T).float())
    print('ANSWER', answ)

# plot_seq([1,0,0,0,0,3,3,3], emb_1_idx)
# plt.savefig('../save/fig/10000555.png')


# <codecell>
### Profiles embeddings
sort_idxs = np.argsort(emb[:,3])
plt.bar(np.arange(256), emb[:,3][sort_idxs])

# %%
### DIMENSIONALITY OF INPUT PROFILES
data = np.copy(emb)

data = StandardScaler().fit_transform(data)
pca = PCA(2)
data_pca = pca.fit_transform(data)

emb_idxs = [emb_0_idx, emb_1_idx, emb_2_idx, emb_5_idx]
names = ['0', '1', '+', 'noop']
top_k = 40

fig, axs = plt.subplots(3, 4, figsize=(20, 12))
pca_axs, cnxn_axs, readout_axs = axs

for emb_idx, name, ax in zip(emb_idxs, names, pca_axs):
    idx_color = np.zeros(256)
    idx_color[emb_idx[-top_k:]] = 1
    # idx_color = emb_idx / 256
    mpb = ax.scatter(data_pca[:,0], data_pca[:,1], c=idx_color, alpha=0.5)
    ax.set_title(f'Token: {name}')
    fig.colorbar(mpb, ax=ax)
    

for emb_idx, name, ax in zip(emb_idxs, names, cnxn_axs):
    w_dest = np.sum([W[:,i] for i in emb_idx[-top_k:]], axis=0)
    mpb = ax.scatter(data_pca[:,0], data_pca[:,1], c=w_dest, alpha=0.5, vmin=-1, vmax=1, cmap='bwr')
    fig.colorbar(mpb, ax=ax)

w_read = model.readout.weight.data.detach().numpy()
for ax in readout_axs:
    mpb = ax.scatter(data_pca[:,0], data_pca[:,1], c=w_read, alpha=0.5, vmin=-0.5, vmax=0.5)
    fig.colorbar(mpb, ax=ax)
    
for ax, name in zip(axs[:,0], ('Sensitivity', 'Connectivity', 'Readout')):
    ax.set_ylabel(name)

fig.tight_layout()
plt.savefig('../save/fig/recurrent_neurons.png')


# idx=emb_2_idx[-50:]
# idx_color = np.zeros(256)
# idx_color[idx] = 1

# plt.scatter(data_pca[:,0], data_pca[:,1], c=idx_color, alpha=0.5)

# w_dest = np.zeros(256)
# for i in idx:
#     w_dest += W[:,i]

# plt.scatter(data_pca[:,0], data_pca[:,1], c=w_dest, alpha=0.5, vmin=-1, vmax=1, cmap='bwr')
# plt.colorbar()



# <codecell>
### INVESTIGATE EMBEDDING DIRECTIONS
data = np.copy(emb)

data = StandardScaler().fit_transform(data)
pca = PCA()
data_pca = pca.fit_transform(data)

plt.plot(np.arange(1, 5), np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, 5))
plt.xlabel('PC')
plt.ylabel('Proportion of explained variance')
plt.title('PCs of neurons in input-sensitivity space')
plt.savefig('../save/fig/pc_input_space.png')

ids = np.identity(4)
pcs = pca.inverse_transform(ids)

print(pcs)


# <codecell>
### POSSIBLE WORKING SOLUTION

W = np.matrix('2,0,0;1,1,-1;1,0,0')
readout = np.matrix('1;1;-1')

tok_to_emb = {
    1: np.matrix('1;0;0'),
    0: np.matrix('0;0;0'),
    2: np.matrix('-100;0;-100')
}

def relu(x): return np.maximum(x, 0)

def predict(seq):
    h = np.zeros((3, 1))
    for tok in seq:
        h = relu(W @ h + tok_to_emb[tok])
    
    return readout.T @ h

predict([1,0,0,0,0,0])


# %%
