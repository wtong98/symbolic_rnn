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


class RNNWithFixedWeights(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = 3
        self._ = torch.nn.Linear(1, 1)

    def forward(self, input_pack, w1, w2):
        dev = next(self.parameters()).device
        W = torch.tensor([
            [0, 0],
            [1, -1],
            [0, 0]
        ]).to(dev)

        W_pre = torch.concat((w1, w2, torch.tensor([1], device=dev))).reshape(-1, 1)
        W = torch.concat((W_pre, W), axis=1)


        data, batch_sizes, _, unsort_idxs = input_pack
        batch_idxs = batch_sizes.cumsum(0)
        batches = torch.tensor_split(data, batch_idxs[:-1])

        hidden = torch.zeros(batch_sizes[0], self.hidden_size).to(dev)
        for b, size in zip(batches, batch_sizes):
            hidden_chunk = torch.relu(b + hidden[:size,...] @ W.T)
            hidden = torch.cat((hidden_chunk, hidden[size:,...]), dim=0)

        hidden = hidden[unsort_idxs]
        return None, hidden.unsqueeze(0)


class RnnClassifier3D(RnnClassifier):
    def __init__(self, fix_emb=False, w1=0, w2=0, **kwargs) -> None:
        nonlinearity = 'relu'
        loss_func = 'mse'
        embedding_size=3
        hidden_size = 3
        vocab_size = 5
        self.fix_emb = fix_emb
        super().__init__(0, nonlinearity=nonlinearity, embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, loss_func=loss_func, **kwargs)

        if fix_emb:
            self.w1 = torch.nn.Parameter(
                torch.tensor([w1]).float(), requires_grad=True
            )

            self.w2 = torch.nn.Parameter(
                torch.tensor([w2]).float(), requires_grad=True
            )

            self.encoder_rnn = RNNWithFixedWeights()

            self.embedding.weight = torch.nn.Parameter(
                torch.tensor([[0,0,0],[1,0,0],[-500,0,-500],[-1,-1,-1], [-1,-1,-1]]).float().reshape(-1, 3), requires_grad=False)
        else:
            self.encoder_rnn.weight_ih_l0 = torch.nn.Parameter(
                torch.eye(3), requires_grad=True
            )

            self.encoder_rnn.bias_ih_l0 = torch.nn.Parameter(
                torch.zeros(3), requires_grad=True
            )

            self.encoder_rnn.bias_hh_l0 = torch.nn.Parameter(
                torch.zeros(3), requires_grad=True
            )

            self.readout.bias = torch.nn.Parameter(
                torch.zeros(1), requires_grad=True
            )

            self.encoder_rnn.weight_hh_l0 = torch.nn.Parameter(
                torch.tensor([[2,0.,0.],[1.,1,-1.],[1.,0.,0.]]), requires_grad=True
            )

            self.readout.weight = torch.nn.Parameter(
                torch.tensor([1,1,-1]).float().reshape(1, -1), requires_grad=True
            )

            self.embedding.weight = torch.nn.Parameter(
                torch.tensor([[0,0,0],[1,0,0],[-500,0,-500],[-1,-1,-1], [-1,-1,-1]]).float().reshape(-1, 3), requires_grad=True)

    def forward(self, x):
        if self.fix_emb:
            hid = self.encode(x, self.w1, self.w2)
            out = self.w2 * hid[:,0] + hid[:,1] - hid[:,2]
            out = out.unsqueeze(1)
        else:
            hid = self.encode(x)
            out = self.readout(hid)
        return out
            
    
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

ds_args_only_test = BinaryAdditionDataset(n_bits=8, 
                           onehot_out=True, 
                           max_args=1, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True,
                           filter_={'min_value':128})


ds_mini = BinaryAdditionDataset(n_bits=2, 
                           onehot_out=True, 
                           max_args=2, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True)

def make_dl(ds, ds_test=None):
    train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds_test if ds_test != None else ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    
    return train_dl, test_dl

# ds_all = ConcatDataset([ds_args_only, ds_full])
# ds_all = ds_args_only
ds_all = ds_full
ds_all.pad_collate = ds_args_only.pad_collate
train_dl, test_dl = make_dl(ds_all, None)

for (x, y), _ in list(zip(ds_all, range(300))):
    print(x.tolist(), y.item())

# print('TEST---')
# for (x, y), _ in list(zip(ds_args_only_test, range(300))):
#     print(x.tolist(), y.item())
# <codecell>
# model = RnnClassifier1D(fix_emb=False, optim=torch.optim.SGD, full_batch=True).cuda()
model = RnnClassifier3D(fix_emb=False, optim=torch.optim.Adam, full_batch=False)
# model = RnnClassifier3D(fix_emb=True, optim=torch.optim.Adam, full_batch=False)
# model.print_params()
model.cuda()
# model.cpu()
# out = model(torch.tensor([[1,2,1,1]]))

# <codecell>
# traj = [(model.encoder_rnn.weight_hh_l0.item(), model.readout.weight.item())]
# model = RnnClassifier3D(fix_emb=True, optim=torch.optim.Adam, w1=0, w2=-1, full_batch=False).cuda()
# traj_1 = [(model.w1.item(), model.w2.item())]
# def eval_cb(model):
    # traj.append((model.encoder_rnn.weight_hh_l0.item(), model.readout.weight.item()))
    # traj_1.append((model.w1.item(), model.w2.item()))
# eval_cb = None

n_epochs = 7000
# losses = model.learn(n_epochs, train_dl, test_dl, lr=1e-3, eval_every=100, eval_cb=eval_cb)
losses = model.learn(n_epochs, train_dl, test_dl, lr=1e-3, eval_every=100)

# print('new mojo')
# model = RnnClassifier3D(fix_emb=True, optim=torch.optim.Adam, w1=-2, w2=-1, full_batch=False).cuda()
# traj_2 = [(model.w1.item(), model.w2.item())]
# def eval_cb2(model):
#     traj_2.append((model.w1.item(), model.w2.item()))
# losses = model.learn(n_epochs, train_dl, test_dl, lr=1e-3, eval_every=100, eval_cb=eval_cb2)

# print('done!')

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
    # axs[0].set_yscale('log')

    axs[1].plot(epochs, losses['train_acc'], label='train acc')
    axs[1].plot(epochs, losses['test_acc'], label='test acc')
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

ds_args_only = BinaryAdditionDataset(n_bits=2, 
                           onehot_out=True, 
                           max_args=2, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True)

print(list(ds_args_only))


def relu(x): return np.maximum(x, 0)


def build_model(w, w_r):
    embedding = {
        0: 0,
        1: 1
    }

    def predict(seq):
        h = 0
        for s in seq:
            h = relu(w * h + embedding[s])
        
        return w_r * h
    
    return predict

def build_model_emb(emb_0, emb_1):
    embedding = {
        0: emb_0,
        1: emb_1
    }

    def predict(seq):
        h = 0
        for s in seq:
            h = relu(2 * h + embedding[s])
        
        return h
    
    return predict

def build_model_2d(w_1, w_2):
    embedding = {
        0: np.array([-1,1]).reshape(-1, 1),
        1: np.array([1,1]).reshape(-1, 1)
    }

    w = np.matrix(f'{w_1},0;,-1,{w_2}')
    readout = np.array([1, -1]).reshape(-1, 1)

    def predict(seq):
        h = np.zeros((2, 1))
        for s in seq:
            h = np.tanh(w @ h + embedding[s])
        
        return readout.T @ h

    return predict

def build_model_3d(w_1, w_2):
    embedding = {
        0: np.array([0,0,0]).reshape(-1, 1),
        1: np.array([1,0,0]).reshape(-1, 1),
        2: np.array([-300, 0, -300]).reshape(-1, 1),
        # 2: np.array([w_1, 0, w_2]).reshape(-1, 1),
    }

    w = np.matrix(f'{w_1},0,0;,{w_2},1,-1;0,0,0')
    readout = np.array([w_2, 1, -1]).reshape(-1, 1)
    # w = np.matrix(f'2,0,0;,1,1,-1;1,0,0')
    # readout = np.array([1, 1, -1]).reshape(-1, 1)

    def predict(seq):
        h = np.zeros((3, 1))
        for s in seq:
            h = relu(w @ h + embedding[s])
        
        return readout.T @ h

    return predict


def get_loss(model, dataset):
    total = 0
    for x, y in dataset:
        pred = model(x.tolist())
        total += (pred - y.item()) ** 2
    
    return total / len(dataset)

def get_loss_manual(model, dataset, w, w_r):
    correct_m = build_model_3d(w, w_r)
    total = 0
    for x, _ in dataset:
        pred = model(x.tolist())
        y = correct_m(x.tolist())
        total += (pred - y) ** 2
    
    return total / len(dataset)



# <codecell>
x = np.arange(-4, 4, 0.3)
y = np.arange(-4, 4, 0.3)

xx, yy = np.meshgrid(x, y)
z = []

for x_, y_ in tqdm(zip(xx.ravel(), yy.ravel()), total=np.prod(xx.shape)):
    m = build_model_3d(x_, y_)
    z.append(get_loss_manual(m, ds_args_only, 2, 1))
    # z.append(get_loss_manual(m, ds_args_only, -300, -300))

z = np.array(z).reshape(xx.shape)

# <codecell>
# TODO: experiment with fixed embeddings for 3D model <-- STOPPED HERE
plt.contourf(xx, yy, np.log(z), 100, vmin=0)
plt.axvline(x=2, alpha=0.3, linestyle='dashed', color='black')
plt.axhline(y=1, alpha=0.3, linestyle='dashed', color='black')
plt.colorbar()
plt.xlabel('Digit multiplier')
plt.ylabel('Accumulator transfer')


traj_1 = np.array(traj_1)
traj_2 = np.array(traj_2)

plt.plot(traj_1[:,0], traj_1[:,1], color='red')
plt.plot(traj_2[:,0], traj_2[:,1], color='red')

plt.savefig('../save/fig/3d_loss_landscape.png')

# plt.xlim((1.5, 2.5))
# plt.ylim((0.5, 1.5))

# <codecell>
### CAMERA-READY PLOTS
fig, axs = plt.subplots(1, 2, figsize=(9, 3))

mpb = axs[0].contourf(xx, yy, np.log(z), 100, vmin=0)
axs[0].axvline(x=2, alpha=0.3, linestyle='dashed', color='black')
axs[0].axhline(y=1, alpha=0.3, linestyle='dashed', color='black')
fig.colorbar(mpb, ax=axs[0])

axs[0].plot(traj[:,0], traj[:,1], color='red')
axs[0].set_xlabel('w')
axs[0].set_ylabel('w_r')

epochs = np.arange(len(losses['train'])) * eval_every
axs[1].plot(epochs, losses['train_acc'], label='train acc')
axs[1].plot(epochs, losses['test_acc'], label='test acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_xlim(0, 4000)
axs[1].legend()

fig.tight_layout()

plt.savefig('../save/fig/w_v_wr_1d_loss_landscape.png')


# <codecell>
traj = np.array(traj)
plt.plot(traj[:,0], traj[:,1], 'o--', alpha=0.5)



# %%
### VALIDATE MODEL
loss_pred = [get_loss(build_model(w, w_r), ds_args_only) for w, w_r in traj]

plt.plot(loss_pred)
plt.plot(losses['test'])
