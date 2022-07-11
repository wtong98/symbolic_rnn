"""
Investigate geometry of model's representations
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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

# %% EXPLORE GEOMETRY
_0_emb, _1_emb, plus_emb, noop_emb = model.get_embedding([0, 1, ds.plus_idx, ds.noop_idx]).T.numpy()
W = model.encoder_rnn.weight_hh_l0


sort_idxs = np.argsort(np.abs(noop_emb))
plot_idxs = np.arange(20)

offsets = [-1, 0, 1]
width = 0.2

plt.gcf().set_size_inches(8, 5)
plt.bar(plot_idxs, noop_emb[sort_idxs][plot_idxs], color='gray', alpha=0.5, label='noop', width=width*3)
plt.bar(plot_idxs - width, _0_emb[sort_idxs][plot_idxs], width=width, label='0', alpha=0.9)
plt.bar(plot_idxs, _1_emb[sort_idxs][plot_idxs], width=width, label='1', alpha=0.9)
plt.bar(plot_idxs+width, plus_emb[sort_idxs][plot_idxs], width=width, label='+', alpha=0.9)
plt.legend()
plt.xticks(plot_idxs)
plt.title('W coordinates by NO-OP magnitude')
plt.xlabel('Coordinate index')
plt.ylabel('Value')

plt.gcf().tight_layout()

# <codecell>
dims = 2
mask = np.zeros(len(_0_emb))
mask[sort_idxs[:dims]] = 1

_0 = torch.tensor(_0_emb * mask).float().reshape(-1, 1)
_1 = torch.tensor(_1_emb * mask).float().reshape(-1, 1)
plus = torch.tensor(plus_emb * mask).float().reshape(-1, 1)
noop_emb = torch.tensor(noop_emb).reshape(-1, 1)

val = torch.tanh(W @ torch.tanh(W @ torch.tanh(_1) + _1) + noop_emb)
out = model.readout(val.T)
print(out)




# <codecell>  REP CLOUD WITH BACKGROUND COLOR
fig, axs = plt.subplots(1, 6, figsize=(18, 3))
# fig, axs = plt.subplots(1, 2, figsize=(8, 3))
mpb = None

# TODO: plot along same PC's?
for n, ax in zip(range(10), axs.ravel()):
    ds = BinaryAdditionDataset(n_bits=3, 
                            onehot_out=True, 
                            max_args=3, 
                            add_noop=True,
                            max_noop=n,
                            max_noop_only=True,
                            #    max_only=True, 
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
    pca = PCA(n_components=2)
    all_points = pca.fit_transform(all_points.T).T

    x = np.linspace(-10, 10, 100)
    xx, yy = np.meshgrid(x, x)
    bg = np.stack((xx.flatten(), yy.flatten()), axis=1)
    with torch.no_grad():
        # some info lost with inverse transform
        hid_bg = torch.tensor(pca.inverse_transform(bg)).float()
        out_labs = model.readout(hid_bg).argmax(dim=-1)
        zz = out_labs.numpy().reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.5)
    mpb = ax.scatter(all_points[0,:], all_points[1,:], c=all_labs_pred)

    ax.set_title(f'n_noops = {n}')

# TODO: what does it look like in 3D?
fig.colorbar(mpb)
fig.tight_layout()
plt.savefig('../save/fig/rnn_noop_cloud_with_background.png')

