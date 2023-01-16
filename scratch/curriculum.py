"""
Exploring curriculum strategies for training effective models

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, Dataset

sys.path.append('../')
from model import *


class RnnClassifier3D(RnnClassifier):
    def __init__(self, embedding_size=3, hidden_size=3, **kwargs) -> None:
        nonlinearity = 'relu'
        loss_func = 'mse'
        embedding_size = embedding_size
        hidden_size = hidden_size
        vocab_size = 6
        super().__init__(0, n_layers=1, nonlinearity=nonlinearity, embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, loss_func=loss_func, **kwargs)

    @torch.no_grad()
    def print_params(self):
        w = self.encoder_rnn.weight_hh_l0.data.numpy()
        emb = self.encoder_rnn.weight_ih_l0 @ self.embedding.weight.T \
            + torch.tile(self.encoder_rnn.bias_ih_l0.reshape(-1, 1), (1, 6)) \
            + torch.tile(self.encoder_rnn.bias_hh_l0.reshape(-1, 1), (1, 6))

        w_r = self.readout.weight.data.numpy()
        b_r = self.readout.bias.data.numpy()

        print('emb_0', emb[:,0].numpy())
        print('emb_1', emb[:,1].numpy())
        print('emb_+', emb[:,2].numpy())
        print('emb_noop', emb[:,-1].numpy())
        print('w', w)
        print('w_r', w_r)
        print('b_r', b_r)


class RnnClassifierBinaryOut(RnnClassifier):
    def __init__(self, n_places=10, nonlinearity='tanh', embedding_size=5, hidden_size=100, n_layers=1, **kwargs) -> None:
        super().__init__(0, nonlinearity=nonlinearity, 
                         use_softexp=False, 
                         embedding_size=embedding_size, 
                         hidden_size=hidden_size, 
                         n_layers=n_layers, **kwargs)
        
        self.n_places = n_places
        self.bitwise_mask = (2 ** torch.arange(n_places - 1, -1, -1)).long()
        self.readout = nn.Linear(self.hidden_size, self.n_places)
        self.loss = nn.BCEWithLogitsLoss()

    def loss(self, logits, targets):
        labels = self.dec_to_bin(targets)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

    # from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    @torch.no_grad()
    def dec_to_bin(self, xs):
        xs = xs.long()
        return xs.unsqueeze(-1) \
                 .bitwise_and(self.bitwise_mask) \
                 .ne(0) \
                 .float()
    
    @torch.no_grad()
    def bin_to_dec(self, xs):
        return torch.sum(self.bitwise_mask * xs, -1)
    
    @torch.no_grad()
    def raw_to_dec(self, xs):
        xs_bin = torch.sigmoid(xs).round()
        return self.bin_to_dec(xs_bin)
    
    @torch.no_grad()
    def pretty_forward(self, xs):
        out = model(xs)
        return self.raw_to_dec(out)
    
    def cuda(self, device = None):
        self.bitwise_mask = self.bitwise_mask.cuda()
        return super().cuda(device)
    
    def cpu(self):
        self.bitwise_mask = self.bitwise_mask.cpu()
        return super().cpu()

class CurriculumDataset(IterableDataset):
    def __init__(self, params, probs=None, max_noops=5) -> None:
        self.params = params
        self.probs = probs
        self.max_noops = max_noops

        self.end_token = '<END>'
        self.pad_token = '<PAD>'
        self.noop_token = '_'
        self.idx_to_token = ['0', '1', '+', self.end_token, self.pad_token, self.noop_token]
        self.token_to_idx = {tok: i for i, tok in enumerate(self.idx_to_token)}

        self.noop_idx = self.token_to_idx[self.noop_token]
        self.plus_idx = self.token_to_idx['+']
    
    def __iter__(self):
        return self
    
    def __next__(self):
        idx = np.random.choice(len(self.params), p=self.probs)
        n_bits, n_args = self.params[idx]
        args = np.random.randint(2 ** n_bits, size=n_args)
        return self.args_to_tok(args, n_bits)

    def args_to_tok(self, args, max_bits):
        toks = '+'.join([f'{a:b}'.zfill(max_bits) + np.random.randint(self.max_noops + 1) * self.noop_token for a in args])
        toks = [self.token_to_idx[t] for t in toks]
        return torch.tensor(toks), torch.tensor(np.sum(args)).float()

    def pad_collate(self, batch):
        xs, ys = zip(*batch)
        pad_id = self.token_to_idx[self.pad_token]
        xs_out = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ys_out = torch.stack(ys)
        return xs_out, ys_out


class CurriculumDatasetTrunc(Dataset):
    def __init__(self, ds, length=1000) -> None:
        ex_iter = iter(ds)
        self.examples = [next(ex_iter) for _ in range(length)]
        self.len = length
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return self.len
     

def build_dl(max_bits=3, max_args=3, batch_size=32, **ds_kwargs):
    params = [(i, j) for i in range(1, max_bits + 1) for j in range(1, max_args + 1)]
    ds = CurriculumDataset(params, **ds_kwargs)
    test_ds = CurriculumDatasetTrunc(ds, length=500)

    train_dl = DataLoader(ds, batch_size=batch_size, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    return train_dl, test_dl

# <codecell>
train_dl, test_dl = build_dl(max_bits=7, max_args=3, max_noops=5, batch_size=256)

# <codecell>
model = RnnClassifier3D(embedding_size=32, hidden_size=256).cuda()
# model = RnnClassifierBinaryOut(n_places=4, embedding_size=32, hidden_size=256).cuda()
losses = model.learn(50, train_dl, test_dl=test_dl, max_iters=5000, eval_every=1, lr=3e-5)

# <codecell>
model.cpu()
for i in range(20):
    ex = [1] + i * [0]
    ex = torch.tensor([ex])
    pred = model(ex)
    print(f'Ex {i}: {pred}')
# %%
model = RnnClassifier3D().cuda()
sched = [3, 4, 5, 6, 7]
test_acc = 0

for max_bits in sched:
    train_dl, test_dl = build_dl(max_bits=max_bits)
    while test_acc < 0.95:
        print('SCHED ', max_bits)
        losses = model.learn(100, train_dl, test_dl=test_dl, max_iters=5000, eval_every=1, lr=3e-5)
        test_acc = losses['test_acc'][-1]
    test_acc = 0

print('done!')

# <codecell>
def make_plots(losses, filename=None, eval_every=100):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    epochs = np.arange(len(losses['train'])) * eval_every
    axs[0].plot(epochs, losses['train'], label='train loss')
    axs[0].plot(epochs, losses['test'], label='test loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # axs[1].plot(epochs, losses['tok_acc'], label='token-wise accuracy')
    # axs[1].plot(epochs, losses['arith_acc'], label='expression-wise accuracy')
    # axs[1].set_xlabel('Epoch')
    # axs[1].set_ylabel('Accuracy')
    # axs[1].legend()

    fig.tight_layout()

    if filename != None:
        plt.savefig(filename)

make_plots(losses, eval_every=1)

# %%  OLD STYLE OF TRAINING
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

def make_dl(ds):
    train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    return train_dl, test_dl

train_dl, test_dl = make_dl(ds_full)

# <codecell>
# model = RnnClassifier3D().cuda()
losses = model.learn(9000, train_dl, test_dl=test_dl, eval_every=100, lr=3e-5)

# %% PLOT SOME EIGNESPECTRA
@torch.no_grad()
def extract_params(model):
    w = model.encoder_rnn.weight_hh_l0.data.numpy()
    emb = model.encoder_rnn.weight_ih_l0 @ model.embedding.weight.T \
        + torch.tile(model.encoder_rnn.bias_ih_l0.reshape(-1, 1), (1, 6)) \
        + torch.tile(model.encoder_rnn.bias_hh_l0.reshape(-1, 1), (1, 6))
    emb = emb.data.numpy()

    w_r = model.readout.weight.data.numpy()
    b_r = model.readout.bias.data.numpy()

    return {
        'w': w,
        'emb': emb,   # symbol x rep
        'w_r': w_r,
        'b_r': b_r
    }

model = RnnClassifier(0)
model.load('save/256d_10bit')

params = extract_params(model)

# <codecell>
vals, vecs = np.linalg.eig(params['w'])
r_idx = np.imag(vals) == 0
sort_idx = np.argsort(-vals[r_idx])

r_vals = vals[np.imag(vals) == 0]
plt.bar(np.arange(len(r_vals)), vals[r_idx][sort_idx])

# %%
w_r = params['w_r']
r_vecs = np.real(vecs[:,r_idx])
r_vecs = r_vecs[:,sort_idx]

score = w_r @ r_vecs
plt.bar(np.arange(18), score.flatten())


# %%
e = params['emb'][:,2].reshape(1, -1)
r_vecs = np.real(vecs[:,r_idx])
r_vecs = r_vecs[:,sort_idx]

score = e @ r_vecs
plt.bar(np.arange(18), score.flatten())

# <codecell>
(w_r @ r_vecs) @ (e @ r_vecs).T

# <codecell>
# TESTING MODEL SAVE / LOAD
model = RnnClassifierBinaryOut()
model.load('save/256d_7bit_bin')
# %%
