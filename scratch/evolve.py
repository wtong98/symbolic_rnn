"""
Explore evolutionary algorithms for training models

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import sys

import cma
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import torch
from tqdm import tqdm

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


ds = BinaryAdditionDataset(n_bits=3, 
                           onehot_out=True, 
                           max_args=3, 
                           use_zero_pad=True,
                           float_labels=True,
                           filter_={
                               'in_args': []
                           })


def args_to_tok(args):
    toks = '+'.join([f'{a:b}' for a in args])
    toks = [ds.token_to_idx[t] for t in toks]
    return torch.tensor(toks), torch.tensor(np.sum(args))

def sample_example(n, arg_low=4, arg_high=7, length_rate=1.25):
    lens = np.random.poisson(length_rate, size=n) + 1
    exs = [np.random.randint(2 ** arg_low, 2 ** arg_high, size=l) for l in lens]
    exs = [args_to_tok(e) for e in exs]
    return ds.pad_collate(exs)

# TODO: add gradient signal to training process
def run(weights, n_steps=10, n_train=32, n_test=32, cuda=False):
    weights = torch.tensor(weights)
    model = RnnClassifier3D(weights).float()

    if cuda:
        model.cuda()

    model.train()
    dl = DataLoader(ds, batch_size=n_train, pin_memory=cuda, collate_fn=ds.pad_collate)
    model.learn(n_steps, dl, lr=1e-4)
    # optimizer = model.optim(model.parameters(), lr=1e-3)
    # for _ in range(n_steps):
    #     x_train, y_train = sample_example(n_train, arg_low=0, arg_high=3)
    #     if cuda:
    #         x_train = x_train.cuda()
    #         y_train = y_train.cuda()

    #         print('X_TRAIN', x_train)
        
    #     optimizer.zero_grad()
    #     model._train_iter(x_train, y_train)
    #     optimizer.step()
        
    
    model.eval()
    with torch.no_grad():
        x_test, y_test = sample_example(n_test, arg_low=4, arg_high=6)
        if cuda:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        preds = model(x_test)
        return torch.mean((preds - y_test) ** 2).item()


conv = []
def cb(xk, convergence):
    global conv
    conv.append(convergence)
    print('CONV', convergence)

# <codecell>

with torch.multiprocessing.Pool(16) as pool:
    res = differential_evolution(run, [(-2.5, 2.5)] * 37, workers=pool.map, maxiter=10, updating='deferred', callback=cb)
# res = cma.fmin(run, np.random.randn(37), sigma0=1, restarts=2)
# %%
run(np.random.randn(37))

# <codecell>
# %timeit run(np.random.randn(37), n_steps=20, cuda=False)


# <codecell>
weights = np.random.randn(37)
n_steps = 1000
n_train = 32
n_test = 32
cuda = True

weights = torch.tensor(weights)
model = RnnClassifier3D(weights).float()

model.cuda()
dl = DataLoader(ds, batch_size=32, pin_memory=True, collate_fn=ds.pad_collate)
model.learn(n_steps, dl)

# model.train()
# optimizer = model.optim(model.parameters(), lr=1e-4)
# for _ in tqdm(range(n_steps)):
#     x_train, y_train = sample_example(n_train, arg_low=0, arg_high=3)
#     if cuda:
#         x_train = x_train.cuda()
#         y_train = y_train.cuda()
    
#     optimizer.zero_grad()
#     logits = model(x_train)
#     loss = model.loss(logits, y_train.double())
#     loss.backward()
#     optimizer.step()
    

model.eval()
with torch.no_grad():
    x_test, y_test = sample_example(n_test, arg_low=0, arg_high=5)
    if cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
    preds = model(x_test)
    out = torch.mean((preds - y_test) ** 2)

out