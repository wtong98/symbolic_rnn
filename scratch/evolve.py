"""
Explore evolutionary algorithms for training models

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from collections import defaultdict
import sys

import cma
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import torch
from tqdm import tqdm

sys.path.append('../')
from model import *

sol_weights = torch.tensor([
    # emb
    0, 0, 0, 
    1, 0, 0, 
    -999, 0, -999, 

    # rnn in
    1, 0, 0, 
    0, 1, 0,
    0, 0, 1,

    # rnn in bias
    0, 0, 0,

    # rnn recur
    2, 0, 0,
    1, 1, -1,
    1, 0, 0,

    # rnn recur bias
    0, 0, 0,

    # readout
    1, 1, -1,

    # readout bias
    0
]).float()

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

def sample_arg(n, arg_low=0, arg_high=4):
    n_bits = np.random.randint(arg_low, arg_high+1, size=n)
    hi = 2 ** n_bits
    lo = 2. ** (n_bits - 1)
    lo[n_bits == 0] = 0
    vals = np.random.randint(lo, hi)
    return vals
    
@torch.no_grad()
def evol_loss(model, n_test, verbose=False, **kwargs):
    # b_fac = 1 - check_b(model, 0) * check_b(model, 1)
    n_fac = check_n(model, 7, loss_penalty=5)
    vocab_fac = check_vocab(model, n_test, arg_low=2, arg_high=5, **kwargs)
    sum_fac = check_sum(model, n_test, **kwargs)
    ex_fac = check_examples(model, n_test)

    l1_fac = np.sum([torch.sum(torch.abs(p)) for p in model.parameters()])
    
    # loss = n_fac + 0.5 * vocab_fac + sum_fac + 2 * ex_fac
    loss = n_fac + 2 * vocab_fac + 2 * ex_fac + 3 * l1_fac

    if verbose:
        return {
            'loss': loss,
            'n_fac': n_fac,
            # 'b_fac': b_fac,
            'vocab_fac': vocab_fac,
            'sum_fac': sum_fac,
            'ex_fac': ex_fac,
            'l1_fac': l1_fac
        }

    else:
        return loss

@torch.no_grad()
def check_b(model, b, atol=0.1):
    pred = model(torch.tensor([[b]])).item()
    return np.isclose(pred, b, atol=atol)

@torch.no_grad()
def check_n(model, n, atol=0.49, loss_penalty=25):
    exs = [args_to_tok([i]) for i in range(n+1)]
    x, y = ds.pad_collate(exs)
    preds = model(x).flatten()
    n_wrong = torch.sum(~torch.isclose(preds, y.float(), atol=atol))

    return n_wrong * loss_penalty

@torch.no_grad()
def check_vocab(model, n_test=32, **kwargs):
    vals = sample_arg(n_test, **kwargs)
    loss = 0

    for b in (0, 1):
        b_vals = 2 * vals + b
        b_vals_out = np.stack((vals, vals, b * np.ones(n_test)), axis=1).astype('int')

        exs_in = [args_to_tok([e]) for e in b_vals]
        exs_in, _ = ds.pad_collate(exs_in)

        exs_out = [args_to_tok(e) for e in b_vals_out]
        exs_out, _ = ds.pad_collate(exs_out)

        model_in = model(exs_in).flatten()
        model_out = model(exs_out).flatten()

        loss += torch.mean((model_in - model_out) ** 2)
    
    return loss

@torch.no_grad()
def check_sum(model, n_test=32, **kwargs):
    loss = 0
    for n_args in (2,):
        vals = sample_arg(n_test * n_args, **kwargs)
        vals = np.split(vals, n_args)
        vals = np.stack(vals, axis=1)

        exs_in = [args_to_tok(e) for e in vals]
        exs_in, _ = ds.pad_collate(exs_in)

        out = [[args_to_tok([e]) for e in vals[:,i]] for i in range(n_args)]
        out = [ds.pad_collate(o)[0] for o in out]

        model_in = model(exs_in).flatten()
        model_out = torch.concat([model(ex) for ex in out], axis=1)
        model_out = torch.sum(model_out, dim=1).flatten()

        loss += torch.mean((model_in - model_out) ** 2)
    
    return loss

@torch.no_grad()
def check_examples(model, n_test=32, atol=0.49, arg_low=0, arg_high=4, **kwargs):
    x_test, y_test = sample_example(n_test, arg_low=arg_low, arg_high=arg_high, **kwargs)
    preds = model(x_test)
    # return torch.sum(~torch.isclose(preds.flatten(), y_test.float(), atol=atol))
    return torch.mean((preds.flatten() - y_test) ** 2)


def run(weights, n_steps=10, n_train=32, n_test=256, cuda=False):
    weights = torch.tensor(weights)
    model = RnnClassifier3D(weights).float()

    if cuda:
        model.cuda()

    # model.train()
    # dl = DataLoader(ds, batch_size=n_train, pin_memory=cuda, collate_fn=ds.pad_collate)
    # model.learn(n_steps, dl, lr=1e-4)

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
    return evol_loss(model, n_test=n_test)
    # with torch.no_grad():
    #     x_test, y_test = sample_example(n_test, arg_low=4, arg_high=6)
    #     if cuda:
    #         x_test = x_test.cuda()
    #         y_test = y_test.cuda()
    #     preds = model(x_test)
    #     # return torch.mean((preds.flatten() - y_test) ** 2).item()
    #     # diffs = (preds.flatten() - y_test) ** 2
    #     # return -torch.sum(diffs < 0.25)
    #     return -torch.sum(torch.isclose(preds.flatten(), y_test.float(), atol=0.5))



conv = []
all_losses = defaultdict(list)

def cb(xk, convergence):
    global conv
    conv.append(convergence)
    print('CONV', convergence)
    if len(conv) % 10 == 0:
        model = RnnClassifier3D(torch.tensor(xk)).float()
        loss = evol_loss(model, n_test=256, verbose=True)
        print('LOSS', loss)

        for key, val in loss.items():
            all_losses[key].append(val)
        
model = RnnClassifier3D(torch.randn(37))
loss = evol_loss(model, 256, verbose=True)
print(loss)


# <codecell>

with torch.multiprocessing.Pool(16) as pool:
    res = differential_evolution(run, [(-2.5, 2.5)] * 37, workers=pool.map, maxiter=2000, updating='deferred', callback=cb)
# res = cma.fmin(run, np.random.randn(37), sigma0=1, restarts=2)

# <codecell>
plt.plot(conv, label='Convergence')
for key, val in all_losses.items():
    val = np.array(val)
    plt.plot(val, label='key')

plt.show()
# <codecell>
# TODO: need negative examples, successor function?
model = RnnClassifier3D(torch.tensor(res.x)).float()
# dl = DataLoader(ds, batch_size=32, pin_memory=False, collate_fn=ds.pad_collate)
# model.learn(20, dl, lr=1e-4)

model(torch.tensor([[1, 0, 0,]]))
# TODO: emphasize vocab <-- STOPPED HERE

evol_loss(model, verbose=True, n_test=256)
# TODO: explore further, consider introducing GD again, how to encode/measure understanding of task?


# %%
run(np.random.randn(37))

# <codecell>
# %timeit run(np.random.randn(37), n_steps=0, n_test=512, cuda=False)


# <codecell>
weights = np.random.randn(37)
n_steps = 0
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
# %%
