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

# <codecell>

ds = BinaryAdditionDataset()

test_split = 0.1
test_len = int(len(ds) * test_split)
train_len = len(ds) - test_len

train_ds, test_ds = random_split(ds, [train_len, test_len])

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)



# <codecell>
### TRAINING
n_epochs = 35

model = BinaryAdditionLSTM(
    embedding_size=2,
    hidden_size=5).cuda()

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

    print(f'Epoch: {e+1}   train_loss: {curr_loss:.4f}   test_loss: {test_loss:.4f}   acc: {test_acc:.4f}')
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

print_test_case(ds, model, [2, 3])
print_test_case(ds, model, [6, 7])
print_test_case(ds, model, ['10', '01'])
print_test_case(ds, model, [5, 25])
print_test_case(ds, model, [16, 16])

# <codecell>
model.save('save/mini')

# %%
