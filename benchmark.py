"""
Measure performance across different models
"""

# <codecell>
from collections import defaultdict, namedtuple
from pathlib import Path
from re import sub

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# <codecell>

from model import *

cached_ds = {}

def test_long_sequence(model, n_start_args=4, n_end_args=10, max_value=9):
    global cached_ds

    n_args = list(range(n_start_args, n_end_args+1))
    all_accs = []

    for n in tqdm(n_args):
        if n in cached_ds:
            dl = cached_ds[n]
        else:
            ds = BinaryAdditionDataset(n_bits=2, max_args=n, onehot_out=True, max_only=True, filter_={'max_value': max_value})
            dl = DataLoader(ds, batch_size=32, pin_memory=True, num_workers=0, collate_fn=ds.pad_collate)
            cached_ds[n] = dl

        _, acc, _ = model.evaluate(dl)
        all_accs.append(acc)
        
    return all_accs, n_args


def test_zero_pad(model):
    pass

def test_skip_arg(model):
    pass

# adapted from https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-97.php
def compress_str(s):
    return '_'.join(
    sub('([A-Z][a-z]+)', r' \1',
    sub('([A-Z]+)', r' \1',
    s.replace('-', ' '))).split()).lower()

def make_plots(losses, filename=None, eval_every=100):
    fig, axs = plt.subplots(1, 2, figsize=(10,4))

    epochs = np.arange(len(losses['train'])) * eval_every
    axs[0].plot(epochs, losses['train'], label='train loss')
    axs[0].plot(epochs, losses['test'], label='test loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, losses['tok_acc'], label='accuracy')
    # axs[1].plot(epochs, losses['arith_acc'], label='expression-wise accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    fig.tight_layout()

    if filename != None:
        plt.savefig(filename)



# <codecell>
# n_iter = 5
# max_value = 9
# n_end_args = 10
# n_epochs = 15000
# eval_every = 100
# optim_lr = 1e-4
n_iter = 1
max_value = 9
n_end_args = 10
arch_width=10
n_epochs = 150
eval_every = 10
optim_lr = 1e-4

fig_dir = Path('save/fig/benchmark')
if not fig_dir.exists():
    fig_dir.mkdir(parents=True)

TestCase = namedtuple('TestCase', ['name', 'model', 'ds', 'n_epochs'])

def make_cases():
    cases = [
        TestCase(name='Flat RNN (full dataset)',
                model=RnnClassifier(max_value, hidden_size=arch_width), 
                ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=False),
                n_epochs=n_epochs),

        TestCase(name='Flat RNN (max args only)',
                model=RnnClassifier(max_value, hidden_size=arch_width), 
                ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=True),
                n_epochs=n_epochs),

        TestCase(name='Flat RNN Reservoir (full dataset)',
                model=ReservoirClassifier(max_value, hidden_size=arch_width), 
                ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=False),
                n_epochs=n_epochs),

        TestCase(name='Flat RNN Reservoir (max args only)',
                model=ReservoirClassifier(max_value, hidden_size=arch_width), 
                ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=True),
                n_epochs=n_epochs),

        TestCase(name='Linear RNN (full dataset)',
                model=LinearRnnClassifier(max_value, hidden_size=arch_width), 
                ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=False),
                n_epochs=n_epochs),

        TestCase(name='Linear RNN (max args only)',
                model=LinearRnnClassifier(max_value, hidden_size=arch_width), 
                ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, max_args=3, max_only=True),
                n_epochs=n_epochs),
    ]
    
    return cases

results = defaultdict(list)
n_args = None

for i in tqdm(range(n_iter)):
    cases = make_cases()
    for case in cases:
        print('Processing:', case.name)
        case.model.cuda()

        ds = case.ds
        dl = DataLoader(ds, batch_size=32, pin_memory=True, num_workers=0, collate_fn=ds.pad_collate)
        losses = case.model.learn(case.n_epochs, dl, dl, logging=False, lr=optim_lr, eval_every=eval_every)
        make_plots(losses, f'{str(fig_dir)}/{compress_str(case.name)}-{i}.png', eval_every=eval_every)

        accs, n_args = test_long_sequence(case.model, n_end_args=n_end_args)
        results[case.name].append(accs)
        

# <codecell>
bw = 0.1
offsets = bw * np.array([-3, -2, -1, 0, 1, 2]) + bw / 2
xs = np.arange(n_end_args - 3)

for (name, result), offset in zip(results.items(), offsets):
    result = np.array(result)
    means = np.mean(result, axis=0)
    serr = np.std(result, axis=0) / np.sqrt(n_iter)

    plt.bar(xs - offset, means, bw, yerr=serr, label=name)

plt.xticks(xs, xs+4)
plt.xlabel('Max number of args')
plt.ylabel('Accuracy')

plt.legend()
plt.savefig(str(fig_dir / 'comparison.png'))

# %%
