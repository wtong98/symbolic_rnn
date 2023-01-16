"""
Visualization routines for progress reports

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from dataclasses import dataclass, field
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

sys.path.append('../../')
from model import *

save_path = Path('../save')


class CurriculumDataset(IterableDataset):
    def __init__(self, params, probs=None, max_noops=5) -> None:
        self.params = params
        self.probs = probs
        self.max_noops = max_noops

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
        toks = '+'.join([f'{a:b}'.zfill(max_bits) + np.random.randint(self.max_noops + 1) * self.noop_token for a in args])
        toks = [self.token_to_idx[t] for t in toks]
        return torch.tensor(toks), torch.tensor(np.sum(args)).float()

    def pad_collate(self, batch):
        xs, ys = zip(*batch)
        pad_id = self.token_to_idx[self.pad_token]
        xs_out = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ys_out = torch.stack(ys)
        return xs_out, ys_out


class RnnClassifierBinaryOut(RnnClassifier):
    def __init__(self, n_places=10, nonlinearity='tanh', embedding_size=5, hidden_size=100, n_layers=1, **kwargs) -> None:
        super().__init__(0, nonlinearity=nonlinearity, 
                         use_softexp=False, 
                         embedding_size=embedding_size, 
                         hidden_size=hidden_size, 
                         n_layers=n_layers, **kwargs)
        
        self.n_places = n_places
        self.bitwise_mask = (2 ** torch.arange(n_places - 1, -1, -1)).long()
        self.readout = nn.Linear(self.hidden_size, self.n_places)
        self.loss = nn.BCEWithLogitsLoss()

    def loss(self, logits, targets):
        labels = self.dec_to_bin(targets)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

    # from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    @torch.no_grad()
    def dec_to_bin(self, xs):
        xs = xs.long()
        return xs.unsqueeze(-1) \
                 .bitwise_and(self.bitwise_mask) \
                 .ne(0) \
                 .float()
    
    @torch.no_grad()
    def bin_to_dec(self, xs):
        return torch.sum(self.bitwise_mask * xs, -1)
    
    @torch.no_grad()
    def raw_to_dec(self, xs):
        xs_bin = torch.sigmoid(xs).round()
        return self.bin_to_dec(xs_bin)
    
    @torch.no_grad()
    def pretty_forward(self, xs):
        out = model(xs)
        return self.raw_to_dec(out)
    
    def cuda(self, device = None):
        self.bitwise_mask = self.bitwise_mask.cuda()
        return super().cuda(device)
    
    def cpu(self):
        self.bitwise_mask = self.bitwise_mask.cpu()
        return super().cpu()

# <codecell>
@dataclass
class Case:
    name: str
    save_path: Path
    n_bits: int
    n_args: int
    max_noops: int = 0
    acc: int = 0
    mse: int = 0
    is_bin: bool = False


max_bits = 10
max_args = 10
n_batches = 4

all_cases = []
for n_bit in range(1, max_bits + 1):
    for n_arg in range(1, max_args + 1):
        all_cases.extend([
            Case(name='Bin 7bit', save_path='256d_7bit_bin', n_bits=n_bit, n_args=n_arg, max_noops=5, is_bin=True),
            Case(name='MSE 3bit', save_path='256d_3bit', n_bits=n_bit, n_args=n_arg, max_noops=5),  # TODO: fix noops?
            Case(name='MSE 7bit', save_path='256d_7bit', n_bits=n_bit, n_args=n_arg, max_noops=5),
            Case(name='MSE 10bit', save_path='256d_10bit', n_bits=n_bit, n_args=n_arg, max_noops=5),
            Case(name='Evo 3bit', save_path='evo/3bit', n_bits=n_bit, n_args=n_arg),
            Case(name='Evo (custom) 3bit', save_path='evo/3bit_custom', n_bits=n_bit, n_args=n_arg),
        ])

for case in tqdm(all_cases):
    if case.is_bin:
        model = RnnClassifierBinaryOut()
    else:
        model = RnnClassifier(0)

    model.load(save_path / case.save_path)

    ds = CurriculumDataset(params=[(case.n_bits, case.n_args)], max_noops=case.max_noops)
    dl = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=ds.pad_collate)
    dl_iter = iter(dl)
    for _ in range(n_batches):
        xs, ys = next(dl_iter)
        with torch.no_grad():
            if case.is_bin:
                preds = model.pretty_forward(xs)
                ys %= 2 ** model.n_places
            else:
                preds = model(xs)

        res = torch.isclose(preds.flatten(), ys, atol=0.5)

        case.acc += torch.mean(res.float()).item() / n_batches
        case.mse += torch.mean((preds.flatten() - ys)**2).item() / n_batches


# %%
df = pd.DataFrame(all_cases)
plot_df = df[(df['name'] == 'Evo 3bit') \
      | (df['name'] == 'Evo (custom) 3bit') \
      | (df['name'] == 'MSE 3bit')]

# <codecell>
g = sns.catplot(data=plot_df, x='n_bits', y='acc', hue='name', col='n_args', col_wrap=3, kind='bar', height=2, aspect=1.5)
g.despine(left=True)
g.legend.set_title("")
plt.savefig('fig/evo_acc_comparison.png')

# %%
plt.clf()
g = sns.catplot(data=plot_df, x='n_bits', y='mse', hue='name', col='n_args', col_wrap=3, kind='bar', height=2, aspect=1.5)
g.despine(left=True)
g.set(yscale='log')
g.legend.set_title("")
plt.savefig('fig/evo_mse_comparison.png')

# <codecell>
plot_df = df[~df['name'].str.contains('^Evo')]
# %%
# <codecell>
g = sns.catplot(data=plot_df, x='n_bits', y='acc', hue='name', col='n_args', col_wrap=3, kind='bar', height=2, aspect=1.5)
g.despine(left=True)
g.legend.set_title("")
plt.savefig('fig/curr_acc_comparison.png')

# %%
plt.clf()
g = sns.catplot(data=plot_df, x='n_bits', y='mse', hue='name', col='n_args', col_wrap=3, kind='bar', height=2, aspect=1.5)
g.despine(left=True)
g.set(yscale='log')
g.legend.set_title("")
plt.savefig('fig/curr_mse_comparison.png')

# %%
