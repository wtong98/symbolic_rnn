"""
Exploring curriculum strategies for training effective models

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset

sys.path.append('../')
from model import *


class RnnClassifier3D(RnnClassifier):
    def __init__(self, weights, **kwargs) -> None:
        nonlinearity = 'relu'
        loss_func = 'mse'
        embedding_size=3
        hidden_size = 3
        vocab_size = 5
        super().__init__(0, nonlinearity=nonlinearity, embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, loss_func=loss_func, **kwargs)

        emb, rnn_in_weight, rnn_in_bias, rnn_rec_weight, rnn_rec_bias, out_weight, out_bias = torch.split(weights, [9, 9, 3, 9, 3, 3, 1])

        emb = emb.reshape(3, 3)
        emb = torch.concat((emb, -torch.ones(2, 3)))
        self.embedding.weight = torch.nn.Parameter(
            emb, requires_grad=True)

        self.encoder_rnn.weight_ih_l0 = torch.nn.Parameter(
            rnn_in_weight.reshape(3, 3), requires_grad=True
        )

        self.encoder_rnn.bias_ih_l0 = torch.nn.Parameter(
            rnn_in_bias, requires_grad=True
        )

        self.encoder_rnn.weight_hh_l0 = torch.nn.Parameter(
            rnn_rec_weight.reshape(3, 3), requires_grad=True
        )

        self.encoder_rnn.bias_hh_l0 = torch.nn.Parameter(
            rnn_rec_bias, requires_grad=True
        )

        self.readout.weight = torch.nn.Parameter(
            out_weight.reshape(1, -1), requires_grad=True
        )

        self.readout.bias = torch.nn.Parameter(
            out_bias, requires_grad=True
        )

    
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


class CurriculumDataset(IterableDataset):
    def __init__(self, params, probs=None, **kwargs) -> None:
        self.params = params
        self.probs = probs

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
        toks = '+'.join([f'{a:b}'.zfill(max_bits) for a in args])
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
     

max_bits = 10
max_args = 3

params = [(i, j) for i in range(1, max_bits + 1) for j in range(1, max_args + 1)]
ds = CurriculumDataset(params)
test_ds = CurriculumDatasetTrunc(ds, length=100)
# next(iter(ds))

train_dl = DataLoader(ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
# %%
model = RnnClassifier3D(torch.randn(37)).cuda()
losses = model.learn(100, train_dl, test_dl=test_dl, max_iters=1000, eval_every=1, lr=3e-5)

# <codecell>
eval_every = 1
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
