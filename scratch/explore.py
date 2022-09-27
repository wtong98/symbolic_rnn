"""
Wide-ranging exploration of model weights, for interpretability
"""

# <codecell>
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
model.load('../save/relu_mse_interleaved_lin_interp')

# <codecell>
### INVESTIGATE DIMENSIONALITY OF NEURAL ACTIVATIONS
with torch.no_grad():
    all_activ = torch.concat([model.encode(x) for x, _ in train_dl], axis=0)
    all_activ = all_activ.numpy()

all_activ = StandardScaler().fit_transform(all_activ)
pca = PCA().fit(all_activ)
plt.plot(np.arange(1, 257), np.cumsum(pca.explained_variance_ratio_), 'o--')
plt.xlim((1, 11))
plt.xticks(np.arange(1, 11)[0::1])

plt.title('Neural activations are low dimensional')
plt.ylabel('Proportion of explained variance')
plt.xlabel('PCs')

plt.savefig('../save/fig/prop_exp_var.png')

# <codecell>
### Which directions do these PCs point?
pc_inv = np.zeros((model.hidden_size, model.hidden_size))
np.fill_diagonal(pc_inv, 1)
pc_inv = pca.inverse_transform(pc_inv)

# sort_idxs = np.argsort(pc_inv[0,:])
# plt.bar(np.arange(model.hidden_size), pc_inv[0,:][sort_idxs])
plt.hist(pc_inv[0,:])

# <codecell>
### How much does each dimension contribute to the final value?
w_readout = model.readout.weight.data.numpy()
# b_readout = model.readout.bias.data.numpy()
w_readout @ pc_inv[0,:].reshape(-1, 1)

plt.bar(np.arange(256), w_readout.flatten())

# <codecell>
### Dimensionality of readout operations
# (Apparently it's quite high)
W = model.hidden.weight.data.numpy()[:256]
w0 = W[8,:].reshape(-1, 1)

norms = np.linalg.norm(W, axis=1).reshape(-1, 1)
sims = W @ w0 / norms / np.linalg.norm(w0)

plt.bar(np.arange(256), np.sort(sims.flatten()))


# %%
