"""
Training a model to perform binary addition

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from model import *

# TODO: missing combo's of mixed frequency args
# TODO: try with larger dataset to highlight generalization difference?
# <codecell>
ds = BinaryAdditionDataset(n_bits=2, little_endian=False, op_filter={
    # 'arg1': [(6, [0, 1, 2, 3])],
    'arg1': [],
    'arg2': [],
})

it = iter(ds)

for _, val in zip(range(40), iter(ds)):
    print(val)


# <codecell>
test_split = 0
test_len = int(len(ds) * test_split)
train_len = len(ds) - test_len

train_ds, test_ds = random_split(ds, [train_len, test_len])
if test_split == 0:
    test_ds = ds

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)

model = BinaryAdditionLSTM(
    embedding_size=5,
    hidden_size=1).cuda()

# <codecell>
### TRAINING
n_epochs = 128000

optimizer = Adam(model.parameters(), lr=3e-4)

losses = {'train': [], 'test': [], 'acc': []}
running_loss = 0
running_length = 0

all_train_loss = []
all_test_loss = []
all_test_tok_acc = []
all_test_arith_acc = []
all_test_arith_acc_no_teacher = []

eval_every = 100

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

    if (e+1) % eval_every == 0:
        curr_loss = running_loss / running_length
        test_loss, test_acc = compute_test_loss(model, test_dl)
        arith_acc_with_teacher, arith_acc_no_teacher = compute_arithmetic_acc(model, test_dl, ds)

        print(f'Epoch: {e+1}   train_loss: {curr_loss:.4f}   test_loss: {test_loss:.4f}   tok_acc: {test_acc:.4f}   arith_acc_with_teacher: {arith_acc_with_teacher:.4f}    arith_acc_no_teacher: {arith_acc_no_teacher:.4f}')
        losses['train'].append(curr_loss)
        losses['test'].append(test_loss)
        losses['acc'].append(test_acc)
        running_loss = 0
        running_length = 0

        all_train_loss.append(curr_loss)
        all_test_loss.append(test_loss)
        all_test_tok_acc.append(test_acc)
        all_test_arith_acc.append(arith_acc_with_teacher)
        all_test_arith_acc_no_teacher.append(arith_acc_no_teacher)


# <codecell>
epochs = (np.arange(0, n_epochs // eval_every) + 1) * eval_every

plt.plot(epochs, all_train_loss, label='train loss')
plt.plot(epochs, all_test_loss, label='test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# plt.savefig('save/fig/micro_loss_curve.png')

# <codecell>
plt.plot(epochs, all_test_tok_acc, label='token-wise accuracy')
plt.plot(epochs, all_test_arith_acc, label='expression-wise accuracy')
plt.plot(epochs, all_test_arith_acc_no_teacher, label='expression-wise accuracy (no teacher)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# plt.savefig('save/fig/micro_acc_curve.png')

# <codecell>
### SIMPLE EVALUATION
model.cpu()

@torch.no_grad()
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

    in_toks = torch.tensor(in_toks)
    out_toks = torch.tensor(out_toks)
    logits, targets = model(in_toks.unsqueeze(0), out_toks.unsqueeze(0))
    pred_seq = logits.numpy().argmax(axis=-1)
    # print(pred_seq)
    try:
        result = ds.tokens_to_args(pred_seq)[0]
    except:
        print('Bad result:', pred_seq)
        return 0

    # seq = torch.tensor(in_toks)
    # pred_seq = model.generate(seq)
    # result = ds.tokens_to_args(pred_seq)[0]

    prefix = 'GOOD '
    if result != np.sum(in_args):
        prefix = 'BAD '

    print('-'*20)
    print(f'{prefix}: {in_args[0]} + {in_args[1]} = {result}')
    print(f'Input:  {in_toks}')
    print(f'Output: {pred_seq.tolist()}')
    print(f'Answer: {out_toks}')
    return result == np.sum(in_args)


#TODO: performs much more poorly without 0 prefix pads, investigate further <-- STOPPED HERE
@torch.no_grad()
def print_test_case_direct(ds, model, in_toks, out_toks):
    in_args = ds.tokens_to_args(in_toks)

    seq = in_toks
    if type(seq) != torch.Tensor:
        seq = torch.tensor(in_toks)

    pred_seq = model.generate(seq)
    print('pred_seq', pred_seq)
    result = ds.tokens_to_args(pred_seq)[0]

    # logits, targets = model(in_toks.unsqueeze(0), out_toks.unsqueeze(0))
    # pred_seq = logits.numpy().argmax(axis=-1)
    # # print(pred_seq)
    # try:
    #     result = ds.tokens_to_args(pred_seq)[0]
    # except:
    #     return 0

    prefix = 'GOOD '
    if result != np.sum(in_args):
        prefix = 'BAD '

    if type(in_toks) != list:
        in_toks = in_toks.tolist()
    if type(out_toks) != list:
        out_toks = out_toks.tolist()

    print('-'*20)
    print(f'{prefix}: {in_args[0]} + {in_args[1]} = {result}')
    print(f'Input:  {in_toks}')
    print(f'Output: {pred_seq.tolist()}')
    print(f'Answer: {out_toks}')
    return result == np.sum(in_args)

total = 0
correct = 0

with torch.no_grad():
    for expression, answer in iter(test_dl):
        for exp, ans in zip(expression, answer):
            while exp[-1] == ds.token_to_idx[ds.pad_token]:
                exp = exp[:-1]
            
            args = ds.tokens_to_args(exp)
            # result = print_test_case(ds, model, args)
            result = print_test_case_direct(ds, model, exp, ans)

            correct += int(result)
            total += 1

print(f'Total acc: {correct / total:.4f}')

# print_test_case(ds, model, [2, 1])
# print_test_case(ds, model, [6, 7])
# print_test_case(ds, model, ['10', '01'])
# print_test_case(ds, model, [5, 25])
# print_test_case(ds, model, [16, 16])

# <codecell>
### INVESTIGATE INFLUENCE OF 0's
total = 0
correct = 0
for a in np.arange(2 ** 6):
    print('a', a)
    total += 1
    result = print_test_case(ds, model, (6, a))
    if result:
        correct += 1

print(f'total acc: {correct / total :.4f}')

# print_test_case_direct(ds, model,
#     [3, 1, 1, 0, 2, 1, 1, 3],
#     [3,3]
# )


# <codecell>
model.save('save/medium_2k')

# %%
### PLOT TRAJECTORIES THROUGH CELL SPACE
all_seqs = []
all_trajs = []

test_seqs = [
    # 0 block
    # [3, 0, 2, 0, 3],
    # [3, 0, 0, 2, 0, 3],
    # [3, 0, 2, 0, 0, 3],
    # [3, 0, 0, 2, 0, 0, 3],

    # 1 block
    # [3, 0, 2, 1, 3],
    # [3, 0, 2, 0, 1, 3],
    # [3, 0, 0, 2, 1, 3],
    # [3, 0, 0, 2, 0, 1, 3],

    # 1 block
    # [3, 1, 2, 0, 3],
    # [3, 0, 1, 2, 0, 3],
    # [3, 1, 2, 0, 0, 3],
    # [3, 0, 1, 2, 0, 0, 3],

    # 2 block
    [3, 1, 2, 1, 3],
    [3, 0, 1, 2, 1, 3],
    [3, 1, 2, 0, 1, 3],
    [3, 0, 1, 2, 0, 1, 3],
    [3, 0, 0, 2, 1, 1, 3],
    [3, 1, 1, 2, 0, 0, 3],

    # 3 block
    # [3, 1, 1, 2, 1, 3],
    # [3, 1, 1, 2, 0, 1, 3],
    # [3, 1, 2, 1, 1, 3],
    # [3, 0, 1, 2, 1, 1, 3],
]

for seq in test_seqs:
    seq = torch.tensor(seq)
    with torch.no_grad():
        info = model.trace(seq)
    
    traj = torch.cat(info['enc']['cell'], axis=1).numpy()
    all_trajs.append(traj)
    all_seqs.append(seq.numpy())

trajs_blob = np.concatenate(all_trajs.copy(), axis=-1)
pca = PCA(n_components=2)
pca.fit_transform(trajs_blob.T)

plt.gcf().set_size_inches(12, 12)
for seq, traj in zip(all_seqs, all_trajs):
    traj = pca.transform(traj.T).T
    jit_x = np.random.uniform() * 0.04
    jit_y = np.random.uniform() * 0.04
    plt.plot(traj[0,:] + jit_x, traj[1,:] + jit_y, 'o-', label=str(seq), alpha=0.8)

plt.legend()
# plt.savefig('save/fig/micro_128k_traj_2.png')

# %%
### PLOT CLOUD OF FINAL CELL STATES BY VALUE
all_points = []
all_labs_true = []
all_labs_pred = []

for seq, out in ds:
    seq = torch.tensor(seq)
    with torch.no_grad():
        info = model.trace(seq)
    
    point = info['enc']['cell'][-1].numpy()
    all_points.append(point)

    lab_true = ds.tokens_to_args(out)
    lab_pred = ds.tokens_to_args(info['out'])
    all_labs_true.append(lab_true[0])
    all_labs_pred.append(lab_pred[0])

all_points = np.concatenate(all_points, axis=-1)
all_points = PCA(n_components=2).fit_transform(all_points.T).T

plt.scatter(all_points[0,:], all_points[1,:], c=all_labs_true)
plt.legend()
# plt.savefig('save/fig/micro_128k_cell_cloud.png')

# <codecell>
plt.scatter(all_points[0,:], all_points[1,:], c=all_labs_pred)
plt.legend()

# %%

