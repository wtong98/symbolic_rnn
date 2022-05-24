"""
Training a model to perform binary addition

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import functools
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader, random_split

from model import *

# <codecell>
class BinaryAdditionDataset(Dataset):  # TODO: make filter
    def __init__(self, n_bits=4, onehot_out=False, max_args = 2, max_only=False, little_endian=False, filter_=None) -> None:
        """
        filter = {
            'max_value': max value representable by expression
        }
        """
        super().__init__()
        self.n_bits = n_bits
        self.onehot_out = onehot_out
        self.max_args = max_args
        self.little_endian = little_endian
        self.filter = filter_ if filter_ != None else {}

        self.end_token = '<END>'
        self.pad_token = '<PAD>'
        self.idx_to_token = ['0', '1', '+', self.end_token, self.pad_token]
        self.token_to_idx = {tok: i for i, tok in enumerate(self.idx_to_token)}

        self.examples = []
        if max_only:
            self.examples.extend(self._exhaustive_enum(self.max_args))
        else:
            for i in (np.arange(max_args) + 1):
                exs = self._exhaustive_enum(i)
                self.examples.extend(exs)
    
    def _exhaustive_enum(self, n_args):
        all_args = []
        for i in (np.arange(self.n_bits) + 1):
            cart_args = i * [[0, 1]]
            args = itertools.product(*cart_args)
            all_args.extend(args)
        
        cart_terms = n_args * [all_args]
        all_terms = itertools.product(*cart_terms)
        
        all_examples = []
        plus_idx = self.token_to_idx['+']
        end_idx = self.token_to_idx[self.end_token]
        for term in all_terms:
            in_toks = functools.reduce(lambda a, b: a + (plus_idx,) + b, term)
            in_toks = (end_idx,) + in_toks + (end_idx,)

            out_val = np.sum(self.tokens_to_args(in_toks))
            if 'max_value' in self.filter and self.filter['max_value'] < out_val:
                continue
            if not self.onehot_out:
                out_val = self.args_to_tokens(out_val, args_only=True)
            all_examples.append((in_toks, out_val))
        
        return all_examples

    def args_to_tokens(self, *args, with_end=True, args_only=False):
        answer = np.sum(args)
        if self.little_endian:
            in_str = '+'.join([f'{a:b}'[::-1] for a in args])
        else:
            in_str = '+'.join([f'{a:b}' for a in args])
        in_toks = [self.end_token] * with_end + list(in_str) + [self.end_token] * with_end
        in_toks = [self.token_to_idx[t] for t in in_toks]

        if args_only:
            return in_toks

        if self.little_endian:
            out_toks = list(f'{answer:b}'[::-1])
        else:
            out_toks = list(f'{answer:b}')
        
        out_toks = [self.end_token] * with_end + out_toks + [self.end_token] * with_end
        out_toks = [self.token_to_idx[t] for t in out_toks]
        return in_toks, out_toks
    
    def tokens_to_args(self, tokens, return_bin=False):
        str_toks = [self.idx_to_token[t] for t in tokens]
        while str_toks[-1] == self.pad_token:
            str_toks = str_toks[:-1]
        if str_toks[0] == self.end_token:
            str_toks = str_toks[1:]
        if str_toks[-1] == self.end_token:
            str_toks = str_toks[:-1]
        
        str_args = ''.join(str_toks).split('+')
        try:
            if self.little_endian:
                args = [int(str_a[::-1], 2) for str_a in str_args]
            else:
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
        xs_out = pad_sequence(xs, batch_first=True, padding_value=pad_id)

        if self.onehot_out:
            ys_out = torch.stack(ys)
        else:
            ys_out = pad_sequence(ys, batch_first=True, padding_value=pad_id)
        return xs_out, ys_out

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.tensor(x), torch.tensor(y)
    
    def __len__(self):
        return len(self.examples)



# <codecell>
# TODO: try without using fixed max args
ds = BinaryAdditionDataset(n_bits=2, 
                           onehot_out=True, 
                           max_args=3, 
                        #    max_only=True, 
                           little_endian=False, filter_={
                               'max_value': 2
})

it = iter(ds)

for _, val in zip(range(300), iter(ds)):
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

# <codecell>
class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_pack):
        dev = next(self.parameters()).device

        data, batch_sizes, _, unsort_idxs = input_pack
        batch_idxs = batch_sizes.cumsum(0)
        batches = torch.tensor_split(data, batch_idxs[:-1])

        hidden = torch.zeros(batch_sizes[0], self.hidden_size).to(dev)
        for b, size in zip(batches, batch_sizes):
            hidden_chunk = self.ih(b) + self.hh(hidden[:size,...])
            hidden = torch.cat((hidden_chunk, hidden[size:,...]), dim=0)

        hidden = hidden[unsort_idxs]
        return None, hidden.unsqueeze(0)

class LinearRnnClassifier(RnnClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.encoder_rnn = LinearRNN(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
        )
    
    def trace(self, input_seq):
        raise NotImplementedError

# TODO: does not seem capable of achieving 100%? <-- STOPPED HERE
model = LinearRnnClassifier(
    max_arg=9,
    embedding_size=5,
    hidden_size=100).cuda()
# model.load('save/hid5_50k_vargs3_rnn_flat_resv')
# model.cuda()
# model.train()

# <codecell>
### TRAINING
n_epochs = 100000
losses = model.learn(n_epochs, train_dl, test_dl, lr=1e-4)

print('done!')

# <codecell>
eval_every = 100
def make_plots(losses, filename=None, eval_every=100):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    epochs = np.arange(len(losses['train'])) * eval_every
    axs[0].plot(epochs, losses['train'], label='train loss')
    axs[0].plot(epochs, losses['test'], label='test loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, losses['tok_acc'], label='token-wise accuracy')
    axs[1].plot(epochs, losses['arith_acc'], label='expression-wise accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    fig.tight_layout()

    if filename != None:
        plt.savefig(filename)

make_plots(losses)

# <codecell>
### SIMPLE EVALUATION
model.cuda()

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
    arg_str = ' + '.join([str(args) for args in in_args])
    print(f'{prefix}: {arg_str} = {result}')
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
        seq = torch.tensor(in_toks, device='cuda')

    pred_seq = model.generate(seq.cuda())
    # result = ds.tokens_to_args(pred_seq)
    result = pred_seq
    result = result[0] if result != None else None

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
    arg_str = ' + '.join([str(args) for args in in_args])
    print(f'{prefix}: {arg_str} = {result}')
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
### MANUAL TRIAL
# total = 0
# correct = 0
# for a in np.arange(2 ** 6):
#     print('a', a)
#     total += 1
#     result = print_test_case(ds, model, (6, a))
#     if result:
#         correct += 1

# print(f'total acc: {correct / total :.4f}')

# print_test_case(ds, model, (1,2,2))

print_test_case_direct(ds, model,
    # [3, 1, 2, 0, 1, 2, 1, 0, 2, 1, 2, 1, 2, 1, 0, 3],
    [3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3],
    [3,3]
)

# print_test_case_direct(ds, model,
#     [3, 1, 0, 2, 0, 0, 0, 0, 1, 3],
#     [3,3]
# )


# <codecell>
model.save('save/hid5_50k_vargs3_rnn_flat_resv')

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
    # [3, 0, 1, 2, 1, 3],
    # [3, 1, 2, 0, 1, 3],
    # [3, 1, 0, 3],
    [3, 0, 1, 0, 3],
    [3, 0, 0, 0, 0, 1, 0, 2, 1, 3],
    # [3, 1, 2, 1, 2, 1, 2, 1, 3],
    # [3, 1, 1, 3]
    # [3, 0, 1, 2, 0, 1, 3],
    # [3, 0, 0, 2, 1, 1, 3],
    # [3, 1, 1, 2, 0, 0, 3],

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
    
    traj = torch.cat(info['enc']['hidden'], axis=1).numpy()
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

# %%
### PLOT CLOUD OF FINAL CELL STATES BY VALUE
all_points = []
all_labs_true = []
all_labs_pred = []

for seq, out in ds:
    seq = torch.tensor(seq)
    with torch.no_grad():
        info = model.trace(seq)
    
    point = info['enc']['hidden'][-1].numpy()
    all_points.append(point)

    # lab_true = ds.tokens_to_args(out)
    # lab_pred = ds.tokens_to_args(info['out'])
    lab_true = [out]
    lab_pred = [info['out']]
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

