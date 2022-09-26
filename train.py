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
from torch.utils.data import DataLoader, random_split, ConcatDataset

from model import *


class BinaryAdditionDataset(Dataset):
    def __init__(self, n_bits=4, onehot_out=False, 
                       max_args = 2, max_only=False, use_zero_pad=True,
                       add_noop=False, max_noop=3, max_noop_only=False,
                       float_labels=False, little_endian=False, filter_=None) -> None:
        """
        filter = {
            'max_value': max value representable by expression
            'in_args': skip any expression with these input args
            'out_args': skip any expression with these output args
        }
        """
        super().__init__()
        self.n_bits = n_bits
        self.onehot_out = onehot_out
        self.max_args = max_args
        self.use_zero_pad = use_zero_pad
        self.add_noop = add_noop
        self.max_noop = max_noop
        self.max_noop_only = max_noop_only
        self.float_labels = float_labels
        self.little_endian = little_endian

        self.filter = {
            'max_value': np.inf,
            'in_args': [],
            'out_args': []
        }

        for k, v in (filter_ or {}).items():
            self.filter[k] = v

        self.end_token = '<END>'
        self.pad_token = '<PAD>'
        self.noop_token = '_'
        self.idx_to_token = ['0', '1', '+', self.end_token, self.pad_token, self.noop_token]
        self.token_to_idx = {tok: i for i, tok in enumerate(self.idx_to_token)}

        self.noop_idx = self.token_to_idx[self.noop_token]
        self.plus_idx = self.token_to_idx['+']

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
            noops = ()

            in_toks = term
            in_toks_tmp = functools.reduce(lambda a, b: a + noops + (plus_idx,) + b, term)
            # in_toks = (end_idx,) + in_toks + (end_idx,)  # END_IDX forcibly removed

            in_args = self.tokens_to_args(in_toks_tmp)
            do_skip = False
            for arg in self.filter['in_args']:
                if arg in in_args:
                    do_skip = True
                    break
            
            # filter out zero starts
            if not self.use_zero_pad:
                for t in term:
                    if t[0] == 0:
                        do_skip = True
                        break
            
            if do_skip:
                continue
                    
            out_val = np.sum(in_args)
            if self.filter['max_value'] < out_val:
                continue
            if out_val in self.filter['out_args']:
                continue
            if not self.onehot_out:
                out_val = self.args_to_tokens(out_val, args_only=True)
            all_examples.append((in_toks, out_val))
        
        return all_examples

    # TODO: unify with _exhautive_enum
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
        
        str_args = ''.join(str_toks).replace('_', '').split('+')
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
        if self.add_noop:
            if self.max_noop_only:
                x = functools.reduce(lambda a, b: a + self.max_noop * (self.noop_idx,) + (self.plus_idx,) + b, x) \
                    + self.max_noop * (self.noop_idx,)
            else:
                x = functools.reduce(lambda a, b: a + np.random.randint(self.max_noop+1) * (self.noop_idx,) + (self.plus_idx,) + b, x) \
                    + np.random.randint(self.max_noop+1) * (self.noop_idx,)
        else:
            x = functools.reduce(lambda a, b: a + (self.plus_idx,) + b, x)

        dtype = torch.float32 if self.float_labels else torch.long
        return torch.tensor(x), torch.tensor(y, dtype=dtype)
    
    def __len__(self):
        return len(self.examples)
# <codecell>
# TODO: try without using fixed max args
ds_full = BinaryAdditionDataset(n_bits=3, 
                           onehot_out=True, 
                           max_args=3, 
                           add_noop=True,
                           max_noop=5,
                           use_zero_pad=True,
                           float_labels=True,
                        #    max_noop_only=True,
                        #    max_only=True, 
                           little_endian=False,
                           filter_={
                               'in_args': []
                           })

ds_args_only = BinaryAdditionDataset(n_bits=7, 
                           onehot_out=True, 
                           max_args=1, 
                           add_noop=True,
                           max_noop=5,
                           use_zero_pad=True,
                           float_labels=True,
                        #    max_noop_only=True,
                        #    max_only=True, 
                           little_endian=False,
                           filter_={
                               'in_args': []
                           })

def make_dl(ds):
    # test_split = 0
    # test_len = int(len(ds) * test_split)
    # train_len = len(ds) - test_len

    # train_ds, test_ds = random_split(ds, [train_len, test_len])
    # if test_split == 0:
    #     test_ds = ds

    train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    return train_dl, test_dl


# <codecell>
# model = NtmClassifier(
#     max_arg=9,
#     embedding_size=5,
#     ctrl_size=100,
#     mem_size=100,
#     word_size=32,
#     vocab_size=6).cuda()

model = RnnClassifier(
    max_arg=0,
    embedding_size=32,
    hidden_size=256,
    vocab_size=6,
    nonlinearity='relu',
    use_softexp=True,
    loss_func='mse').cuda()

# model.load('save/hid100k_vargs3_nbits3')

# <codecell>
### TRAINING
# train_dl, test_dl = make_dl(ds_args_only)
# n_epochs = 5000
# losses = model.learn(n_epochs, train_dl, test_dl, lr=5e-5, eval_every=100)

# model.fix_ewc(train_dl)
# print(model.old_params)

# train_dl, test_dl = make_dl(ds_full)
# n_epochs = 200
# losses = model.learn(n_epochs, train_dl, test_dl, lr=5e-5, eval_every=100)

ds_all = ConcatDataset([ds_args_only, ds_full])
ds_all.pad_collate = ds_args_only.pad_collate
train_dl, test_dl = make_dl(ds_all)

print(list(zip(ds_all, range(300))))

# <codecell>
n_epochs = 10000
losses = model.learn(n_epochs, train_dl, test_dl, lr=2e-5, eval_every=100)

# model.fix_ewc(train_dl)
# print(model.old_params)

# n_epochs = 200
# losses = model.learn(n_epochs, train_dl, test_dl, lr=5e-5, eval_every=100)


print('done!')
# model.save('save/ntm_nbits_3')

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
    arg_str = ' + '.join([str(args) for args in in_args])
    print(f'{prefix}: {arg_str} = {result}')
    print(f'Input:  {in_toks}')
    print(f'Output: {pred_seq.tolist()}')
    print(f'Answer: {out_toks}')
    return result == np.sum(in_args)


@torch.no_grad()
def print_test_case_direct(ds, model, in_toks, out_toks):
    in_args = ds.tokens_to_args(in_toks)

    seq = in_toks
    if type(seq) != torch.Tensor:
        seq = torch.tensor(in_toks)

    pred_seq = model.generate(seq)
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

# print_test_case_direct(ds, model,
#     # [3, 1, 2, 0, 1, 2, 1, 0, 2, 1, 2, 1, 2, 1, 0, 3],
#     # [1, 5, 2, 1, 5, 5, 5, 2, 1, 5, 2, 1, 5, 5, 2, 1, 2, 5, 5, 1],
#     [1, 2, 1, 5, 5, 5, 2, 1, 2, 1, 5, 5, 2, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
#     [3,3]
# )

# TODO: try with soft exp on expanded dataset <-- STOPPED HERE
# TODO: understand traj of relu_nbits3_nozeropad
print_test_case_direct(ds, model,
    [1, 0, 0, 0, 0, 0, 0] + 0 * [5],
    [3,3]
)


# %%
# model.save('save/relu_nbits3_nozeropad')

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
    [0,0,0,0,1,0,2,1],
    [1,1],
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
    traj = np.concatenate((np.zeros((2, 1)), traj), axis=-1)
    jit_x = np.random.uniform() * 0.04
    jit_y = np.random.uniform() * 0.04
    plt.plot(traj[0,:] + jit_x, traj[1,:] + jit_y, 'o-', label=str(seq), alpha=0.8)

plt.annotate('enc start', (traj[0,0], traj[1,0]))
plt.legend()
# plt.savefig('save/fig/micro_128k_traj_2.png')

# %%
model.save('save/relu_mse_interleaved_lin_interp')

# %%
### PLOT CLOUD OF FINAL CELL STATES BY VALUE

fig, axs = plt.subplots(1, 6, figsize=(18, 3))
mpb = None

W = model.encoder_rnn.weight_hh_l0.data.numpy()
# W = model.encoder_rnn.hh.weight.detach().numpy()
pca = PCA(n_components=2)
pca.fit(W)

# TODO: plot along same PC's?
for n, ax in zip(range(10), axs.ravel()):
    # n *= 20
    ds = BinaryAdditionDataset(n_bits=3, 
                            onehot_out=True, 
                            use_zero_pad=True,
                            max_args=3, 
                            add_noop=True,
                            max_noop=n,
                            max_noop_only=True,
                            #    max_only=True, 
                            little_endian=False)

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
    # all_points = PCA(n_components=2).fit_transform(all_points.T).T
    all_points = pca.transform(all_points.T).T

    mpb = ax.scatter(all_points[0,:], all_points[1,:], c=all_labs_true)
    ax.set_title(f'n_noops = {n}')

# TODO: formally measure embedding dim of activations
fig.colorbar(mpb)
fig.tight_layout()
plt.savefig('save/fig/rnn_relu_softexp_nbits3_mse.png')

# %%
pca.explained_variance_ratio_
# <codecell>
plt.scatter(all_points[0,:], all_points[1,:], c=all_labs_pred)
plt.legend()

# %%
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

for i in range(max(all_labs_true)+1):
    idx = np.array(all_labs_true) == i
    plt.scatter(all_points[0,:][idx], all_points[1,:][idx], label=str(i))

plt.legend()
# plt.savefig('save/fig/100_hid_cell_cloud.png')

# <codecell>
for i in range(max(all_labs_pred)+1):
    idx = np.array(all_labs_pred) == i
    plt.scatter(all_points[0,:][idx], all_points[1,:][idx], label=str(i))

plt.legend()

# <codecell>
# INVESTIGATE GEOMETRY

# TODO: perhaps not super close to 0?
embs = model.embedding(torch.tensor([ds.noop_idx]))
# embs = model.embedding(torch.tensor([4]))
embs = model.encoder_rnn.weight_ih_l0 @ embs.T + model.encoder_rnn.bias_ih_l0.unsqueeze(1) + model.encoder_rnn.bias_hh_l0.unsqueeze(1)
embs = embs.flatten().detach().numpy()

sort_idxs = np.argsort(np.abs(embs))
plot_idxs = np.arange(20)

embs = model.embedding(torch.tensor([ds.plus_idx]))
# embs = model.embedding(torch.tensor([4]))
embs = model.encoder_rnn.weight_ih_l0 @ embs.T + model.encoder_rnn.bias_ih_l0.unsqueeze(1) + model.encoder_rnn.bias_hh_l0.unsqueeze(1)
embs = embs.flatten().detach().numpy()
plt.bar(plot_idxs, embs[sort_idxs][plot_idxs])
# model.encoder_rnn.weight_hh_l0 @ embs

# %%