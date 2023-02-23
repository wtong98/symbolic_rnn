"""
Measure performance across different models
"""

# <codecell>
from dataclasses import dataclass
import pickle

from collections import defaultdict, namedtuple
from pathlib import Path
from re import sub

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *

@dataclass
class Case:
    name: str
    save_path: Path
    max_n_bits_train: int
    max_n_args_train: int
    n_bits_test: int
    n_args_test: int
    max_iters: int = 5000
    n_epochs: int = 5
    max_noops: int = 5
    acc: int = 0
    is_bin: bool = False
    embedding_size: int = 512
    hidden_size: int = 512
    is_mse: bool = False
    max_arg: int = None
    fix_noop: bool = False

root = Path('tmp_models')

def plot_losses(losses, filename=None, eval_every=100):
    epochs = np.arange(len(losses['train'])) * eval_every
    plt.plot(epochs, losses['train'], label='train loss')
    plt.plot(epochs, losses['test'], label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if filename != None:
        plt.savefig(filename)


def build_dl(max_bits=3, max_args=3, batch_size=32, **ds_kwargs):
    params = [(i, j) for i in range(1, max_bits + 1) for j in range(1, max_args + 1)]
    ds = CurriculumDataset(params, **ds_kwargs)
    test_ds = CurriculumDatasetTrunc(ds, length=500)

    train_dl = DataLoader(ds, batch_size=batch_size, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    return train_dl, test_dl


def run_case(case: Case, n_batches=4, force_refresh=False, debug=False):
    if case.is_bin:
        model = RnnClassifierBinaryOut(n_places=4, embedding_size=case.embedding_size, hidden_size=case.hidden_size)
    elif case.is_mse:
        model = RnnClassifier(0, nonlinearity='relu', embedding_size=case.embedding_size, hidden_size=case.hidden_size, loss_func='mse')
    else:
        case.max_arg = (2 ** case.max_n_bits_train - 1) * case.max_n_args_train
        model = RnnClassifier(case.max_arg, embedding_size=case.embedding_size, hidden_size=case.hidden_size)
    
    if Path(root / case.save_path).exists() and force_refresh == False:
        model.load(root / case.save_path)
    else:
        model.cuda()
        train_dl, test_dl = build_dl(max_bits=case.max_n_bits_train, max_args=case.max_n_args_train, max_noops=case.max_noops, batch_size=256)
        losses = model.learn(case.n_epochs, train_dl, test_dl, max_iters=case.max_iters, eval_every=1, lr=5e-5)

        save_path = Path(root / case.save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        model.save(save_path)
        plot_losses(losses, eval_every=1, filename=str(root / case.save_path / 'losses.png'))
        plt.clf()
    
    if debug:
        case.model = model
    
    # eval
    model.cpu()
    ds = CurriculumDataset(params=[(case.n_bits_test, case.n_args_test)], max_noops=case.max_noops, max_out=case.max_arg, fix_noop=case.fix_noop)
    dl = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=ds.pad_collate)
    dl_iter = iter(dl)
    for _ in range(n_batches):
        xs, ys = next(dl_iter)
        with torch.no_grad():
            if case.is_bin:
                preds = model.pretty_forward(xs)
                ys %= 2 ** model.n_places
            elif case.is_mse:
                preds = model(xs)
            else:
                logits = model(xs)
                preds = torch.argmax(logits, axis=1).float()

        res = torch.isclose(preds.flatten(), ys, atol=0.5)

        case.acc += torch.mean(res.float()).item() / n_batches


# case = Case('test', save_path='test_mse', max_n_args_train=3, max_n_bits_train=7, n_bits_test=8, n_args_test=3, n_epochs=10, max_iters=5000, is_mse=True)
# run_case(case, debug=True, force_refresh=False)
# print('done!')

# case

# <codecell>
max_bits = 10
n_args = [1, 3]

# TODO: prepare for cluster and run <-- STOPPED HERE
n_epochs = 10
n_iters = 4
max_iters_per_epoch = 5000

# n_epochs = 3
# n_iters = 2
# max_iters_per_epoch = 50

all_cases = []
for i in range(n_iters):
    for n_bit in range(1, max_bits + 1):
        for n_arg in n_args:
            all_cases.extend([
                Case('Bin 7bit', save_path=f'bin_7bit_{i}', max_n_args_train=3, max_n_bits_train=7, n_bits_test=n_bit, n_args_test=n_arg, n_epochs=n_epochs, max_iters=max_iters_per_epoch, is_bin=True),
                Case('MSE 3bit', save_path=f'mse_3bit_{i}', max_n_args_train=3, max_n_bits_train=3, n_bits_test=n_bit, n_args_test=n_arg, n_epochs=n_epochs, max_iters=max_iters_per_epoch, is_mse=True),
                Case('MSE 7bit', save_path=f'mse_7bit_{i}', max_n_args_train=3, max_n_bits_train=7, n_bits_test=n_bit, n_args_test=n_arg, n_epochs=n_epochs, max_iters=max_iters_per_epoch, is_mse=True),
                Case('MSE 7bit single', save_path=f'mse_7bit_single_{i}', max_n_args_train=1, max_n_bits_train=7, n_bits_test=n_bit, n_args_test=n_arg, n_epochs=n_epochs, max_iters=max_iters_per_epoch, is_mse=True),
            ])


for case in tqdm(all_cases):
    run_case(case)

df = pd.DataFrame(all_cases)
df.to_pickle(root / 'df.pkl')
'''

# <codecell>
df = pd.read_pickle(root / 'df.pkl')

# <codecell>
### Single arg generalization
# plot_df = df[~df['name'].str.contains('single')]
plot_df = df
plot_df = plot_df[plot_df['n_args_test'] == 1]
g = sns.barplot(plot_df, x='n_bits_test', y='acc', hue='name')

g.legend().set_title('')

# <codecell>
### Three arg generalization
plot_df = df[~df['name'].str.contains('single')]
plot_df = plot_df[plot_df['n_args_test'] == 3]
g = sns.barplot(plot_df, x='n_bits_test', y='acc', hue='name')

g.legend().set_title('')


# <codecell>
### Eigenvalues
names = []
max_eigvals = []

for _, row in df.iloc[:4].iterrows():
    if row['is_bin'] == True:
        model = RnnClassifierBinaryOut()
    else:
        model = RnnClassifier(0)
    
    model.load(root / row['save_path'])
    W = model.encoder_rnn.weight_hh_l0.detach()
    eigvals = np.linalg.eigvals(W)
    eigvals = eigvals[np.imag(eigvals) == 0]
    max_eigval = np.real(np.max(eigvals))

    names.append(row['name'])
    max_eigvals.append(max_eigval)

# <codecell>
plt.bar(np.arange(4), max_eigvals, color=['green', 'red', 'red', 'red'])  # TODO: tune colors
plt.xticks(np.arange(4))
plt.gca().set_xticklabels(names)



# <codecell>
### OLD STUFF vvv
def test_long_sequence(model, n_start_args=4, n_end_args=10, max_value=21, add_noop=True):
    global cached_ds

    n_args = list(range(n_start_args, n_end_args+1))
    all_accs = []

    for n in tqdm(n_args):
        if (n, add_noop) in cached_ds:
            dl = cached_ds[(n, add_noop)]
        else:
            ds = BinaryAdditionDataset(n_bits=3, max_args=n, onehot_out=True, max_only=True, add_noop=add_noop, max_noop=5, max_noop_only=True, filter_={'max_value': max_value})
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
        plt.clf()



# <codecell>
n_iter = 5
arch_width = 256
emb_width = 32
max_value = 21
n_end_args = 7
n_epochs = 1000
eval_every = 200
optim_lr = 1e-4


fig_dir = Path('save/fig/benchmark_noop_cosyne')
if not fig_dir.exists():
    fig_dir.mkdir(parents=True)

TestCase = namedtuple('TestCase', ['name', 'model', 'ds', 'n_epochs', 'test_params'])

def make_cases():
    cases = [
        TestCase(name='Full dataset',
                model=RnnClassifier(max_value, hidden_size=arch_width, embedding_size=emb_width), 
                ds=BinaryAdditionDataset(n_bits=3, onehot_out=True, add_noop=True, max_noop=5, max_args=3, max_only=False),
                n_epochs=n_epochs,
                test_params={}),

        TestCase(name='Max args only',
                model=RnnClassifier(max_value, hidden_size=arch_width, embedding_size=emb_width), 
                ds=BinaryAdditionDataset(n_bits=3, onehot_out=True, add_noop=True, max_noop=5, max_args=3, max_only=True),
                n_epochs=n_epochs,
                test_params={}),

        TestCase(name='No noops',
                model=RnnClassifier(max_value, hidden_size=arch_width, embedding_size=emb_width), 
                ds=BinaryAdditionDataset(n_bits=3, onehot_out=True, add_noop=False, max_noop=0, max_args=3, max_only=False),
                n_epochs=n_epochs,
                test_params={'add_noop': False}),

        # TestCase(name='Flat RNN Reservoir (full dataset)',
        #         model=ReservoirClassifier(max_value, hidden_size=arch_width), 
        #         ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, add_noop=True, max_noop=5, max_args=3, max_only=False),
        #         n_epochs=n_epochs),

        # TestCase(name='Flat RNN Reservoir (max args only)',
        #         model=ReservoirClassifier(max_value, hidden_size=arch_width), 
        #         ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, add_noop=True, max_noop=5, max_args=3, max_only=True),
        #         n_epochs=n_epochs),

        # TestCase(name='RNN + MLP (full dataset)',
        #         model=RnnClassifierWithMLP(max_value, hidden_size=arch_width), 
        #         ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, add_noop=True, max_noop=5, max_args=3, max_only=False),
        #         n_epochs=n_epochs),

        # TestCase(name='RNN + MLP (max args only)',
        #         model=RnnClassifierWithMLP(max_value, hidden_size=arch_width), 
        #         ds=BinaryAdditionDataset(n_bits=2, onehot_out=True, add_noop=True, max_noop=5, max_args=3, max_only=True),
        #         n_epochs=n_epochs),
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
        # make_plots(losses, f'{str(fig_dir)}/{compress_str(case.name)}-{i}.png', eval_every=eval_every)

        accs, n_args = test_long_sequence(case.model, n_start_args=1, n_end_args=n_end_args, **case.test_params)
        results[case.name].append(accs)
        

# <codecell>
# TODO: save cases
with open('benchmark_out.pk', 'wb') as fp:
    pickle.dump(results, fp)

# <codecell>
with open('benchmark_out.pk', 'rb') as fp:
    results = pickle.load(fp)
# <codecell>
bw = 0.2
# offsets = bw * np.array([-3, -2, -1, 0, 1, 2]) + bw / 2
offsets = bw * np.array([-1, 0, 1]) + bw / 2
xs = np.arange(n_end_args)

plt.gcf().set_size_inches(7, 3)
plt.ylim((0, 1.1))

for (name, result), offset in zip(results.items(), offsets):
    result = np.array(result)
    means = np.mean(result, axis=0)
    serr = 2 * np.std(result, axis=0) / np.sqrt(n_iter)

    plt.bar(xs - offset, means, bw, yerr=serr, label=name)

plt.axvline(x=2.4, color='k', linestyle='dashed')
plt.annotate('Test split', (2.45, 1.0))

plt.xticks(xs, xs + 1)
plt.xlabel('Max number of args')
plt.ylabel('Accuracy')

plt.legend(bbox_to_anchor=(0.7, 0.7))
plt.savefig('viz/cosyne_fig/comparison.svg', bbox_inches='tight')
# plt.clf()

# %%
'''