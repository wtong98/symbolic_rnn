"""
Plotting generalization trend of models on single argument datasets
"""

# <codecell>
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')

from model import *

def run_case(model_params, ds_params_set, n_epochs=1000, n_zeros=20):
    model = RnnClassifier(**model_params).cuda()
    results = []

    for params in ds_params_set:
        ds = BinaryAdditionDataset(**params)
        train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
        test_dl = DataLoader(ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
        model.learn(n_epochs, train_dl, test_dl, lr=5e-5, eval_every=100)

    print('testing')
    with torch.no_grad():
        model.cpu()
        for i in range(n_zeros+1):
            example = [1] + i * [0]
            ans = model(torch.tensor([example])).item()
            results.append(ans)
    
    print('done')
    return results

Case = namedtuple('Case', 'name model_params ds_params_set results')

common_model_args = {
    'max_arg': 0,
    'embedding_size': 32,
    'hidden_size': 256,
    'vocab_size': 6,
    'use_softexp': True,
    'loss_func': 'mse',
}

common_ds_args = {
    'onehot_out': True, 
    'add_noop': True,
    'max_noop': 5,
    'use_zero_pad': True,
    'float_labels': True
}

case_set = [
    Case(name='Full dataset', 
         model_params=dict({'nonlinearity': 'relu'}, **common_model_args, ),
         ds_params_set=[
            dict({'n_bits': 3, 'max_args': 3}, **common_ds_args)
         ], results=[]),

    Case(name='Single-args only', 
         model_params=dict({'nonlinearity': 'relu'}, **common_model_args),
         ds_params_set=[
            dict({'n_bits': 7, 'max_args': 1}, **common_ds_args)
         ], results=[]),

    Case(name='Single-args, then full', 
         model_params=dict({'nonlinearity': 'relu'}, **common_model_args, ),
         ds_params_set=[
            dict({'n_bits': 7, 'max_args': 1}, **common_ds_args),
            dict({'n_bits': 3, 'max_args': 3}, **common_ds_args)
         ], results=[]),

    Case(name='Full, then single-args', 
         model_params=dict({'nonlinearity': 'relu'}, **common_model_args, ),
         ds_params_set=[
            dict({'n_bits': 3, 'max_args': 3}, **common_ds_args),
            dict({'n_bits': 7, 'max_args': 1}, **common_ds_args)
         ], results=[]),
]

for c in case_set:
    results = run_case(c.model_params, c.ds_params_set)
    c.results.extend(results)

# %%
xs = np.arange(21)
ys = 2 ** xs

plt.plot(xs, ys, 'o--', color='black', label='True')
plt.xticks(xs[::2])
plt.axvline(x=7, color='red')
# plt.annotate('train', (5, 2))
# plt.annotate('test', (7.5, 2))

for c in case_set:
    plt.plot(xs, c.results, 'o--', label=c.name, alpha=0.7)

plt.yscale('log', base=2)
plt.legend()
plt.xlabel('Number of zeros')
plt.ylabel('Numeric value')

plt.savefig('../save/fig/zeros_extrapolation.png')

# %%
