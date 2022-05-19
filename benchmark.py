"""
Measure performance across different models
"""

# <codecell>
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# <codecell>

from model import *

def test_long_sequence(model, n_start_args=4, n_end_args=10):
    n_args = list(range(n_start_args, n_end_args+1))
    all_accs = []
    for n in n_args:
        ds = BinaryAdditionDataset(max_args=n, max_only=True)
        dl = DataLoader(ds, batch_size=32, pin_memory=True, num_workers=0, collate_fn=ds.pad_collate)

        try:
            acc, _ = compute_arithmetic_acc(model, dl)
        except:
            acc = compute_arithmetic_acc_flat(model, dl)
        
        all_accs.append(acc)
        
    return all_accs, n_args


def test_zero_pad(model):
    pass

def test_skip_arg(model):
    pass


# <codecell>
n_iter = 5

TestCase = namedtuple('TestCase', ['name', 'model', 'train_ds', 'n_epochs'])

cases = [
    TestCase(name='Flat RNN (full dataset)',
             model=BinaryAdditionFlatRNN(9), 
             ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=False),
             n_epochs=128000),

    TestCase(name='Flat RNN (max args only)',
             model=BinaryAdditionFlatRNN(9), 
             ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=True),
             n_epochs=128000),

    TestCase(name='Flat RNN Reservoir (full dataset)',
             model=BinaryAdditionFlatReservoirRNN(9), 
             ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=False),
             n_epochs=128000),

    TestCase(name='Flat RNN Reservoir (max args only)',
             model=BinaryAdditionFlatReservoirRNN(9), 
             ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=True),
             n_epochs=128000),
]

# TODO: save loss curves
rnn_lin_dec = BinaryAdditionFlatRNN(9)

