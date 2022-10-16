"""
Exploring the smallest possible NNs that perfectly implement binary addition.

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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

# ds_all = ConcatDataset([ds_args_only, ds_full])
ds_all = ds_args_only
ds_all.pad_collate = ds_args_only.pad_collate
train_dl, test_dl = make_dl(ds_all)

for (x, y), _ in list(zip(ds_all, range(300))):
    print(x.tolist())

# <codecell>
model = RnnClassifier(
    max_arg=0,
    embedding_size=32,
    hidden_size=1,
    vocab_size=6,
    nonlinearity='relu',
    use_softexp=False,
    l1_weight=0,
    loss_func='mse').cuda()

# <codecell>
n_epochs = 10000
losses = model.learn(n_epochs, train_dl, test_dl, lr=2e-5, eval_every=100)
print('done!')

# <codecell>
eval_every = 100
def make_plots(losses, filename=None, eval_every=100):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    epochs = np.arange(len(losses['train'])) * eval_every
    axs[0].plot(epochs, losses['train'], label='train loss')
    axs[0].plot(epochs, losses['test'], label='test loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, losses['tok_acc'], label='token-wise accuracy')
    axs[1].plot(epochs, losses['arith_acc'], label='expression-wise accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    fig.tight_layout()

    if filename != None:
        plt.savefig(filename)

make_plots(losses)

# <codecell>
### Manually visualize loss landscape
ds_args_only = BinaryAdditionDataset(n_bits=7, 
                           onehot_out=True, 
                           max_args=1, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True)

def relu(x): return np.maximum(x, 0)

embedding = {
    0: 0,
    1: 1
}

def build_model(w, w_r):
    def predict(seq):
        h = 0
        for s in seq:
            h = relu(w * h + embedding[s])
        
        return w_r * h
    
    return predict

def get_loss(model, dataset):
    total = 0
    for x, y in dataset:
        pred = model(x.tolist())
        total += (pred - y.item()) ** 2
    
    return total / len(dataset)


x = np.arange(1, 3, 0.1)
y = np.arange(0, 2, 0.1)

xx, yy = np.meshgrid(x, y)
z = []

for x_, y_ in tqdm(zip(xx.ravel(), yy.ravel())):
    model = build_model(x_, y_)
    z.append(get_loss(model, ds_args_only))

z = np.array(z).reshape(xx.shape)

# <codecell>
# TODO: try training model with fixed embedding, and plot trajectory
plt.contourf(xx, yy, np.log(np.log(z)), 100)
plt.axvline(x=2, alpha=0.3, linestyle='dashed')
plt.axhline(y=1, alpha=0.3, linestyle='dashed')
plt.colorbar()







# %%
