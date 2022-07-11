"""
Find the critical / slow points of the system, and dynamics between them
"""
# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch

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


# <codecell>
def tanh_deriv(x):
    return 1 - torch.tanh(x) ** 2

@torch.no_grad()
def compute_jacob(model, h, tok_idx):
    h = torch.tensor(h).reshape(-1, 1).float()
    emb = model.get_embedding([tok_idx])
    W = model.encoder_rnn.weight_hh_l0
    F = -h + torch.tanh(W @ h + emb)

    deriv = tanh_deriv(W @ h + emb).tile(1, W.shape[1])
    J = -torch.eye(W.shape[1]) + deriv * W
    return F, J

@torch.no_grad()
def compute_grads(model, h, tok_idx):
    F, J = compute_jacob(model, h, tok_idx)

    q = 0.5 * torch.norm(F) ** 2
    grad = J @ F
    hess = J @ J.T
    return q, grad, hess

@torch.no_grad()
def make_funcs(model, tok_idx):
    def q(x): return compute_grads(model, x, tok_idx)[0].numpy()
    def jac(x): return compute_grads(model, x, tok_idx)[1].flatten().numpy()
    def hess(x): return compute_grads(model, x, tok_idx)[2].numpy()

    return q, jac, hess


# <codecell>
info = model.trace([1, 5, 5, 5, 2, 1, 5, 5, 5])
q, jac, hess = make_funcs(model, 5)
h_start = info['enc']['hidden'][3].detach().numpy()
res = minimize(q, h_start, method='bfgs', jac=jac, hess=hess, options={'disp': True})

# NOTE: some funny business with Hessian



# %%
