"""
Training a model to perform binary addition

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from model import *

# TODO: try with little-endian style
class BinaryAdditionDataset(Dataset):
    def __init__(self, max_len=None, n_bits=4) -> None:
        super().__init__()
        self.end_token = '<END>'
        self.pad_token = '<PAD>'
        self.idx_to_token = ['0', '1', '+', self.end_token, self.pad_token]
        self.token_to_idx = {tok: i for i, tok in enumerate(self.idx_to_token)}

        if max_len != None:
            self.examples = [self.args_to_tokens(
                np.random.randint(0, 2 ** n_bits),
                np.random.randint(0, 2 ** n_bits)
            ) for _ in range(max_len)]
        else:
            self.examples = self._exhaustive_enum(n_bits)
    
    def _exhaustive_enum(self, n_bits):
        all_examples = []

        for i in (np.arange(n_bits) + 1):
            for a in range(2 ** i):
                for b in range(2 ** i):
                    in_toks, out_toks = self.args_to_tokens(a, b)
                    a_str, b_str = ''.join([str(t) for t in in_toks])[1:-1].split(str(self.token_to_idx['+']))

                    a_str = '0' * (i - len(a_str)) + a_str
                    b_str = '0' * (i - len(b_str)) + b_str
                    in_str = a_str + '+' + b_str
                    in_toks = [self.end_token] + list(in_str) + [self.end_token]

                    in_toks = [self.token_to_idx[t] for t in in_toks]
                    all_examples.append((in_toks, out_toks))
        
        return all_examples
    
    def args_to_tokens(self, *args):
        answer = np.sum(args)
        in_str = '+'.join([f'{a:b}' for a in args])
        in_toks = [self.end_token] + list(in_str) + [self.end_token]
        out_toks = [self.end_token] + list(f'{answer:b}') + [self.end_token]

        in_toks = [self.token_to_idx[t] for t in in_toks]
        out_toks = [self.token_to_idx[t] for t in out_toks]
        return in_toks, out_toks
    
    def tokens_to_args(self, tokens, return_bin=False):
        str_toks = [self.idx_to_token[t] for t in tokens]
        if str_toks[0] == self.end_token:
            str_toks = str_toks[1:]
        if str_toks[-1] == self.end_token:
            str_toks = str_toks[:-1]
        
        str_args = ''.join(str_toks).split('+')
        try:
            args = [int(str_a, 2) for str_a in str_args]
        except:
            # print('invalid tokens: ', tokens)
            return None

        if return_bin:
            return args, str_args
        else:
            return args
    
    def pad_collate(self, batch):
        xs, ys = zip(*batch)
        pad_id = self.token_to_idx[self.pad_token]
        xs_pad = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ys_pad = pad_sequence(ys, batch_first=True, padding_value=pad_id)
        return xs_pad, ys_pad

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.tensor(x), torch.tensor(y)
    
    def __len__(self):
        return len(self.examples)


# <codecell>

ds = BinaryAdditionDataset(n_bits=6)

test_split = 0.1
test_len = int(len(ds) * test_split)
train_len = len(ds) - test_len

train_ds, test_ds = random_split(ds, [train_len, test_len])

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=16, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)

model = BinaryAdditionLSTM(
    embedding_size=5,
    hidden_size=100).cuda()

# <codecell>
### TRAINING
n_epochs = 100

optimizer = Adam(model.parameters(), lr=3e-4)

losses = {'train': [], 'test': [], 'acc': []}
running_loss = 0
running_length = 0

for e in range(n_epochs):
    for i, (input_seq, output_seq) in enumerate(train_dl):
        input_seq = input_seq.cuda()
        output_seq = output_seq.cuda()

        optimizer.zero_grad()

        logits, targets = model(input_seq, output_seq)
        loss = model.loss(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_length += 1

    curr_loss = running_loss / running_length
    test_loss, test_acc = compute_test_loss(model, test_dl)
    arith_acc = compute_arithmetic_acc(model, test_dl, ds)

    print(f'Epoch: {e+1}   train_loss: {curr_loss:.4f}   test_loss: {test_loss:.4f}   tok_acc: {test_acc:.4f}   arith_acc: {arith_acc:.4f}')
    losses['train'].append(curr_loss)
    losses['test'].append(test_loss)
    losses['acc'].append(test_acc)
    running_loss = 0

# <codecell>
### SIMPLE EVALUATION
model.cpu()

def print_test_case(ds, model, args):
    in_args = []
    was_binary = False
    for a in args:
        if type(a) == str:  # assumed to be a binary str
            a = int(a, 2)
            was_binary = True

        in_args.append(a)

    in_toks, out_toks = ds.args_to_tokens(*in_args)
    if was_binary:
        end_idx = ds.token_to_idx[ds.end_token]
        in_toks = list('+'.join(args))
        in_toks = [end_idx] + [ds.token_to_idx[t] for t in in_toks] + [end_idx]

    seq = torch.tensor(in_toks)
    pred_seq = model.generate(seq)
    result = ds.tokens_to_args(pred_seq)[0]

    prefix = 'GOOD '
    if result != np.sum(in_args):
        prefix = 'BAD '

    print('-'*20)
    print(f'{prefix}: {in_args[0]} + {in_args[1]} = {result}')
    print(f'Input:  {in_toks}')
    print(f'Output: {pred_seq.tolist()}')
    print(f'Answer: {out_toks}')

print_test_case(ds, model, [2, 1])
print_test_case(ds, model, [6, 7])
print_test_case(ds, model, ['10', '01'])
print_test_case(ds, model, [5, 25])
print_test_case(ds, model, [16, 16])

# <codecell>
model.save('save/mini')

# <codecell>
model.load('save/prototype')

# %%
