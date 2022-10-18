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


class RnnClassifier1D(RnnClassifier):
    def __init__(self, fix_emb=False, **kwargs) -> None:
        nonlinearity = 'relu'
        loss_func = 'mse'
        embedding_size=1
        hidden_size = 1
        vocab_size = 5
        super().__init__(0, nonlinearity=nonlinearity, embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, loss_func=loss_func, **kwargs)

        if fix_emb:
            self.embedding.weight = torch.nn.Parameter(
                torch.tensor([0., 1., -1., -1., -1.]).reshape(-1, 1), requires_grad=False)
            
            self.encoder_rnn.weight_ih_l0 = torch.nn.Parameter(
                torch.tensor([[1.]]), requires_grad=False
            )

            self.encoder_rnn.bias_ih_l0 = torch.nn.Parameter(
                torch.tensor([0.]), requires_grad=False
            )

            self.encoder_rnn.bias_hh_l0 = torch.nn.Parameter(
                torch.tensor([0.]), requires_grad=False
            )

            self.readout.bias = torch.nn.Parameter(
                torch.tensor([0.]), requires_grad=False
            )
    
    def print_params(self):
        w = self.encoder_rnn.weight_hh_l0.data.item()
        emb = self.encoder_rnn.weight_ih_l0 @ self.embedding.weight.T + self.encoder_rnn.bias_ih_l0 + self.encoder_rnn.bias_hh_l0

        w_r = self.readout.weight.data.item()
        b_r = self.readout.bias.data.item()

        print('emb_0', emb[0,0].item())
        print('emb_1', emb[0,1].item())
        print('w', w)
        print('w_r', w_r)
        print('b_r', b_r)


class RnnClassifier3D(RnnClassifier):
    def __init__(self, fix_emb=False, **kwargs) -> None:
        nonlinearity = 'relu'
        loss_func = 'mse'
        embedding_size=3
        hidden_size = 3
        vocab_size = 5
        super().__init__(0, nonlinearity=nonlinearity, embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, loss_func=loss_func, **kwargs)

        if fix_emb:
            pass
    
    @torch.no_grad()
    def print_params(self):
        w = self.encoder_rnn.weight_hh_l0.data.numpy()
        emb = self.encoder_rnn.weight_ih_l0 @ self.embedding.weight.T \
            + torch.tile(self.encoder_rnn.bias_ih_l0.reshape(-1, 1), (1, 5)) \
            + torch.tile(self.encoder_rnn.bias_hh_l0.reshape(-1, 1), (1, 5))

        w_r = self.readout.weight.data.numpy()
        b_r = self.readout.bias.data.numpy()

        print('emb_0', emb[:,0].numpy())
        print('emb_1', emb[:,1].numpy())
        print('emb_+', emb[:,2].numpy())
        print('w', w)
        print('w_r', w_r)
        print('b_r', b_r)



# <codecell>
ds_full = BinaryAdditionDataset(n_bits=3, 
                           onehot_out=True, 
                           max_args=3, 
                        #    add_noop=True,
                        #    max_noop=5,
                           use_zero_pad=True,
                           float_labels=True,
                        #    max_noop_only=True,
                        #    max_only=True, 
                           little_endian=False,
                           filter_={
                               'in_args': []
                           })

# ds_args_only = BinaryAdditionDataset(n_bits=7, 
#                            onehot_out=True, 
#                            max_args=1, 
#                            add_noop=True,
#                            max_noop=5,
#                            use_zero_pad=True,
#                            float_labels=True,
#                         #    max_noop_only=True,
#                         #    max_only=True, 
#                            little_endian=False,
#                            filter_={
#                                'in_args': []
#                            })

ds_args_only = BinaryAdditionDataset(n_bits=7, 
                           onehot_out=True, 
                           max_args=1, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True)

def make_dl(ds):
    train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    return train_dl, test_dl

ds_all = ConcatDataset([ds_args_only, ds_full])
# ds_all = ds_args_only
ds_all.pad_collate = ds_args_only.pad_collate
train_dl, test_dl = make_dl(ds_all)

for (x, y), _ in list(zip(ds_all, range(300))):
    print(x.tolist())

# <codecell>
# model = RnnClassifier1D(fix_emb=False, optim=torch.optim.SGD, full_batch=True).cuda()
# model = RnnClassifier1D(fix_emb=False, optim=torch.optim.Adam, full_batch=False).cuda()
model = RnnClassifier3D(fix_emb=False, optim=torch.optim.Adam, full_batch=False)
model.print_params()
model.cuda()


# <codecell>
# traj = [(model.encoder_rnn.weight_hh_l0.data.item(), model.readout.weight.data.item())]
# def eval_cb(model):
#     traj.append((model.encoder_rnn.weight_hh_l0.data.item(), model.readout.weight.data.item()))
eval_cb = None

n_epochs = 10000
losses = model.learn(n_epochs, train_dl, test_dl, lr=1e-4, eval_every=100, eval_cb=eval_cb)
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
# ds_args_only = BinaryAdditionDataset(n_bits=7, 
#                            onehot_out=True, 
#                            max_args=1, 
#                            add_noop=False,
#                            use_zero_pad=True,
#                            float_labels=True)

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

def get_loss_manual(model, dataset, w, w_r):
    correct_m = build_model(w, w_r)
    total = 0
    for x, _ in dataset:
        pred = model(x.tolist())
        y = correct_m(x.tolist())
        total += (pred - y) ** 2
    
    return total / len(dataset)



# <codecell>
x = np.arange(-1, 3, 0.1)
y = np.arange(-1, 3, 0.1)

xx, yy = np.meshgrid(x, y)
z = []

for x_, y_ in tqdm(zip(xx.ravel(), yy.ravel()), total=np.prod(xx.shape)):
    m = build_model(x_, y_)
    z.append(get_loss(m, ds_args_only))

z = np.array(z).reshape(xx.shape)

# <codecell>
# TODO: experiment with fixed embeddings for 3D model <-- STOPPED HERE
plt.contourf(xx, yy, np.log(z), 100)
plt.axvline(x=2, alpha=0.3, linestyle='dashed')
plt.axhline(y=1, alpha=0.3, linestyle='dashed')
plt.colorbar()

# traj = np.array(traj)
# plt.plot(traj[:,0], traj[:,1], 'o--', alpha=0.5, color='black', markersize=2)

# plt.xlim((1.5, 2.5))
# plt.ylim((0.5, 1.5))



# <codecell>
traj = np.array(traj)
plt.plot(traj[:,0], traj[:,1], 'o--', alpha=0.5)



# %%
### VALIDATE MODEL
loss_pred = [get_loss(build_model(w, w_r), ds_args_only) for w, w_r in traj]

plt.plot(loss_pred)
plt.plot(losses['test'])
