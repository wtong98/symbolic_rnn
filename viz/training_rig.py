"""
Plotting generalization trend of models on single argument datasets
"""

# <codecell>
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import ConcatDataset

import sys
sys.path.append('../')

from model import *

def run_case(model_params, ds_params_set, n_epochs=1000, n_zeros=20, interleaved=False):
    model = RnnClassifier(**model_params).cuda()
    results = []

    if interleaved:
        all_ds = [BinaryAdditionDataset(**params) for params in ds_params_set]
        ds = ConcatDataset(all_ds)
        ds.pad_collate = all_ds[0].pad_collate

        train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
        test_dl = DataLoader(ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
        model.learn(n_epochs, train_dl, test_dl, lr=5e-5, eval_every=100)
    else:
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

Case = namedtuple('Case', 'name model_params ds_params_set is_interleaved results', defaults=[False, []])

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
    Case(name='Interleaved', 
         model_params=dict({'nonlinearity': 'relu'}, **common_model_args, ),
         ds_params_set=[
            dict({'n_bits': 3, 'max_args': 3}, **common_ds_args),
            dict({'n_bits': 7, 'max_args': 1}, **common_ds_args)
         ], is_interleaved=True),
]

for c in case_set:
    results = run_case(c.model_params, c.ds_params_set, interleaved=c.is_interleaved)
    c.results.extend(results)

# <codecell>
# TODO: save out-data
with open('cosyne_fig/zeros_bench_out.pk', 'wb') as fp:
    pickle.dump(case_set, fp)

# <codecell>
with open('cosyne_fig/zeros_bench_out.pk', 'rb') as fp:
    case_set = pickle.load(fp)

# %%
xs = np.arange(21)
ys = 2 ** xs

plt.gcf().set_size_inches(7, 4.3)

plt.plot(xs, ys, 'o--', color='black', label='True')
plt.xticks(xs[::2])
plt.axvline(x=7, color='red', alpha=0.8)
plt.axvline(x=2, color='magenta', alpha=0.8)
plt.annotate('Full split', (2.1, 150), color='magenta')
plt.annotate('Single split', (7.2, 0.6), color='red')

names = {i: case_set[i].name for i in range(len(case_set))}
names[0] = 'Full'
names[1] = 'Single'
names[2] = 'Single, then full'
names[3] = 'Full, then single'
names[4] = 'Interleaved'

for i, c in enumerate(case_set):
    offset = 0
    if i == 0:
        offset = 2
    plt.plot(xs, np.array(c.results) + offset, 'o--', label=names[i], alpha=0.6)

plt.yscale('log', base=2)
plt.legend()
plt.xlabel('Number of zeros')
plt.ylabel('Numeric value')

plt.savefig('cosyne_fig/zeros_extrapolation.svg')

# %%
