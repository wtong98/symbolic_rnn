"""
Measure performance across different models
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np

# <codecell>

from model import *

def test_long_sequence(model, n_start_args=4, n_end_args=10):
    n_args = list(range(n_start_args, n_end_args+1))
    for n in n_args:
        ds = BinaryAdditionDataset(max_args=n, max_only=True)
        # TODO: track accuracy
    



def test_zero_pad(model):
    pass

def test_skip_arg(model):
    pass