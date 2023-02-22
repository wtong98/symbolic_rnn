"""
Generate plots for COSYNE
"""

# <codecell>
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm

import sys
sys.path.append('../')
from model import *

# <codecell>
### Point clouds
def plot_point_cloud(model, title='', save_path=None):
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    # fig, axs = plt.subplots(4, 1, figsize=(2, 8))
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
        ax.set_title(f'noops = {n}')
        ax.axis('off')
        ax.grid(True)
        # ax.patch.set_edgecolor('gray')
        # ax.patch.set_linewidth('0.5')

    fig.colorbar(mpb)
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)

# <codecell>
model = RnnClassifier(0)
model.load('../save/hid100k_vargs3_nbits3')
plot_point_cloud(model, save_path='cosyne_fig/cloud_full.svg')

# <codecell>
model = RnnClassifier(0)
model.load('../save/skip_6')
plot_point_cloud(model, save_path='cosyne_fig/cloud_skip_6.svg')

# <codecell>
model = RnnClassifier(0, loss_func='mse', nonlinearity='tanh')
model.load('../save/relu_mse_interleaved')
plot_point_cloud(model, save_path='cosyne_fig/cloud_relu_mse.svg')

# <codecell>
# NOTE: zero's extrapolation relegated to `training_rig.py`
# NOTE: length extrapolation relegated to `benchmark.py`
# TRAJECTORY PLOT

def tanh_deriv(x):
    return 1 - torch.tanh(x) ** 2

@torch.no_grad()
def compute_jacob(model, h, tok_idx):
    h = torch.tensor(h).reshape(-1, 1).float()
    emb = model.get_embedding([tok_idx])
    W = model.encoder_rnn.weight_hh_l0
    F = -h + torch.tanh(W @ h + emb)

    deriv = tanh_deriv(W @ h + emb).tile(1, W.shape[1])
    J = -torch.eye(W.shape[1]) + deriv * W
    return F, J

@torch.no_grad()
def compute_grads(model, h, tok_idx):
    F, J = compute_jacob(model, h, tok_idx)

    q = 0.5 * torch.norm(F) ** 2
    grad = J @ F
    hess = J @ J.T
    return q, grad, hess

@torch.no_grad()
def make_funcs(model, tok_idx):
    def q(x): return compute_grads(model, x, tok_idx)[0].numpy()
    def jac(x): return compute_grads(model, x, tok_idx)[1].flatten().numpy()
    def hess(x): return compute_grads(model, x, tok_idx)[2].numpy()

    return q, jac, hess

@torch.no_grad()
def optim(model, seq, h_init_idx, tok_idx):
    info = model.trace(seq)
    q, jac, hess = make_funcs(model, tok_idx)
    h_start = info['enc']['hidden'][h_init_idx].detach().numpy()
    res = minimize(q, h_start, method='trust-ncg', jac=jac, hess=hess)

    if not res['success']:
        res = minimize(q, h_start, method='bfgs', jac=jac)

    return res

def find_crit_points(model, all_seqs):
    all_results = []
    for seq in all_seqs:
        curr_results = []
        print('Compute seq:', seq)
        for i in tqdm(range(len(seq))):
            res = optim(model, seq, i, 5)
            curr_results.append(res)
        all_results.append(curr_results)
    
    return all_results

def plot_seq(model, all_seqs, all_results=None, save_path=None):
    all_trajs = []
    all_labs = []
    if all_results == None:
        all_results = [[]]  # TODO: temporary

    W = model.encoder_rnn.weight_hh_l0.data.numpy()
    pca = PCA(n_components=2)
    pca.fit(np.linalg.matrix_power(W, 1))

    for seq in all_seqs:
        seq = torch.tensor(seq)
        with torch.no_grad():
            info = model.trace(seq)

        traj = torch.cat(info['enc']['hidden'], axis=-1)
        labs = model.readout(traj.T).argmax(axis=-1)
        all_labs.append(labs.tolist())

        traj = traj.numpy()
        traj = pca.transform(traj.T).T
        all_trajs.append(traj)

    plt.gcf().set_size_inches(6, 6)
    lss = ['-', ':']
    for seq, traj, labs, result, ls in zip(all_seqs, all_trajs, all_labs, all_results, lss):
        # traj = pca.transform(traj.T).T
        traj = np.concatenate((np.zeros((2, 1)), traj), axis=-1)
        # jit_x = np.random.uniform(-1, 1) * 0.05
        # jit_y = np.random.uniform(-1, 1) * 0.05
        jit_x = 0
        jit_y = 0

        diffs = traj[:,1:] - traj[:,:-1]
        for i, point, lab, diff in zip(range(len(seq)), traj.T, labs, diffs.T):
            if seq[i] == 5:
                color = 'C0'
            elif seq[i] == 1:
                color = 'C1'
            elif seq[i] == 0:
                color = 'C4'
            else:
                color = 'C2'
            
            point[0] += jit_x
            point[1] += jit_y
            plt.arrow(*point, *diff, color=color, alpha=0.8, length_includes_head=True, width=0.03, head_width=0.25, head_length=0.2)
            # plt.annotate(lab, point + 0.1)
        
        # plt.plot(traj[0,:] + jit_x, traj[1,:] + jit_y, 'o-', label=str(seq), alpha=0.8)


        for i, res in enumerate(result):
            if res['success']:
                crit_pt = pca.transform(res['x'].reshape(1, -1))
                plt.plot(crit_pt[0,0], crit_pt[0,1], 'ro', markersize=7, alpha=0.6)

                label = model.readout(torch.tensor(res['x'].reshape(1, -1)).float()).argmax(dim=-1)
                if label.item() <= 8:
                    offset = 0.15
                else:
                    offset = np.array([-0.45, 0.35]).reshape(2, 1)

                plt.annotate(label.item(), crit_pt.T + offset, color='red')

                # orig_pt = traj[:,i]
                # plt.arrow(orig_pt[0], orig_pt[1], crit_pt[0,0] - orig_pt[0], crit_pt[0,1] - orig_pt[1], alpha=0.2)

    plt.annotate('Start', (0.12, -0.1), color='red')
    plt.axis('off')


    handles = [
        mpatches.Patch(color='C0', label='No-Op'),
        mpatches.Patch(color='C1', label='1'),
        mpatches.Patch(color='C4', label='0'),
        mpatches.Patch(color='C2', label='+'),
        mpatches.Patch(color='red', label='Crit. Pt')
    ]

    # plt.legend(handles=handles)
    if save_path:
        plt.savefig(save_path)


# <codecell>
model = RnnClassifier(0)
model.load('../save/hid100k_vargs3_nbits3')
all_seqs = [[1,5,5,5] + 15 * [2,1,5,5,5]]
all_results = find_crit_points(model, all_seqs)
plot_seq(model, all_seqs, all_results, save_path='cosyne_fig/traj_full_no_legend.svg')

# <codecell>
model = RnnClassifier(0)
model.load('../save/skip_6')
all_seqs = [[1,5,5,5,2,1,1,0,5,5,5,2,1,5,5,5]]
all_results = find_crit_points(model, all_seqs)
plot_seq(model, all_seqs, all_results, save_path='cosyne_fig/traj_skip_6.svg')


# <codecell>
### 1D RNN loss landscape
class RnnClassifier1D(RnnClassifier):
    def __init__(self, fix_emb=False, **kwargs) -> None:
        nonlinearity = 'relu'
        loss_func = 'mse'
        embedding_size=1
        hidden_size = 1
        vocab_size = 5
        super().__init__(0, nonlinearity=nonlinearity, embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, loss_func=loss_func, **kwargs)

        self.encoder_rnn.weight_hh_l0 = torch.nn.Parameter(
            torch.tensor([[1.]]), requires_grad=True
        )

        self.readout.weight = torch.nn.Parameter(
            torch.tensor([[1.78]]), requires_grad=True
        )

        if fix_emb:
            self.embedding.weight = torch.nn.Parameter(
                torch.tensor([0., 1., -1., -1., -1.]).reshape(-1, 1), requires_grad=False)
            
            self.encoder_rnn.weight_ih_l0 = torch.nn.Parameter(
                torch.tensor([[1.]]), requires_grad=False
            )

            self.encoder_rnn.bias_ih_l0 = torch.nn.Parameter(
                torch.tensor([0.]), requires_grad=False
            )

            self.encoder_rnn.bias_hh_l0 = torch.nn.Parameter(
                torch.tensor([0.]), requires_grad=False
            )

            self.readout.bias = torch.nn.Parameter(
                torch.tensor([0.]), requires_grad=False
            )
    
    def print_params(self):
        w = self.encoder_rnn.weight_hh_l0.data.item()
        emb = self.encoder_rnn.weight_ih_l0 @ self.embedding.weight.T + self.encoder_rnn.bias_ih_l0 + self.encoder_rnn.bias_hh_l0

        w_r = self.readout.weight.data.item()
        b_r = self.readout.bias.data.item()

        print('emb_0', emb[0,0].item())
        print('emb_1', emb[0,1].item())
        print('w', w)
        print('w_r', w_r)
        print('b_r', b_r)


class RNNWithFixedWeights(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = 3
        self._ = torch.nn.Linear(1, 1)

    def forward(self, input_pack, w1, w2):
        dev = next(self.parameters()).device
        W = torch.tensor([
            [0, 0],
            [1, -1],
            [0, 0]
        ]).to(dev)

        W_pre = torch.concat((w1, w2, torch.tensor([1], device=dev))).reshape(-1, 1)
        W = torch.concat((W_pre, W), axis=1)


        data, batch_sizes, _, unsort_idxs = input_pack
        batch_idxs = batch_sizes.cumsum(0)
        batches = torch.tensor_split(data, batch_idxs[:-1])

        hidden = torch.zeros(batch_sizes[0], self.hidden_size).to(dev)
        for b, size in zip(batches, batch_sizes):
            hidden_chunk = torch.relu(b + hidden[:size,...] @ W.T)
            hidden = torch.cat((hidden_chunk, hidden[size:,...]), dim=0)

        hidden = hidden[unsort_idxs]
        return None, hidden.unsqueeze(0)


class RnnClassifier3D(RnnClassifier):
    def __init__(self, fix_emb=False, w1=0, w2=0, **kwargs) -> None:
        nonlinearity = 'relu'
        loss_func = 'mse'
        embedding_size=3
        hidden_size = 3
        vocab_size = 5
        self.fix_emb = fix_emb
        super().__init__(0, nonlinearity=nonlinearity, embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, loss_func=loss_func, **kwargs)

        if fix_emb:
            self.w1 = torch.nn.Parameter(
                torch.tensor([w1]).float(), requires_grad=True
            )

            self.w2 = torch.nn.Parameter(
                torch.tensor([w2]).float(), requires_grad=True
            )

            self.encoder_rnn = RNNWithFixedWeights()

            self.embedding.weight = torch.nn.Parameter(
                torch.tensor([[0,0,0],[1,0,0],[-500,0,-500],[-1,-1,-1], [-1,-1,-1]]).float().reshape(-1, 3), requires_grad=False)
            

    def forward(self, x):
        if self.fix_emb:
            hid = self.encode(x, self.w1, self.w2)
            out = self.w2 * hid[:,0] + hid[:,1] - hid[:,2]
            out = out.unsqueeze(1)
        else:
            hid = self.encode(x)
            out = self.readout(hid)
        return out
            
    
    @torch.no_grad()
    def print_params(self):
        w = self.encoder_rnn.weight_hh_l0.data.numpy()
        emb = self.encoder_rnn.weight_ih_l0 @ self.embedding.weight.T \
            + torch.tile(self.encoder_rnn.bias_ih_l0.reshape(-1, 1), (1, 5)) \
            + torch.tile(self.encoder_rnn.bias_hh_l0.reshape(-1, 1), (1, 5))

        w_r = self.readout.weight.data.numpy()
        b_r = self.readout.bias.data.numpy()

        print('emb_0', emb[:,0].numpy())
        print('emb_1', emb[:,1].numpy())
        print('emb_+', emb[:,2].numpy())
        print('w', w)
        print('w_r', w_r)
        print('b_r', b_r)


ds_full = BinaryAdditionDataset(n_bits=3, 
                           onehot_out=True, 
                           max_args=3, 
                           use_zero_pad=True,
                           float_labels=True,
                           little_endian=False,
                           filter_={
                               'in_args': []
                           })


ds_args_only = BinaryAdditionDataset(n_bits=7, 
                           onehot_out=True, 
                           max_args=1, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True)

ds_args_only_test = BinaryAdditionDataset(n_bits=8, 
                           onehot_out=True, 
                           max_args=1, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True,
                           filter_={'min_value':128})

ds_args_only_mini_test = BinaryAdditionDataset(n_bits=4, 
                           onehot_out=True, 
                           max_args=1, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True,
                           filter_={'min_value':8})


ds_mini = BinaryAdditionDataset(n_bits=2, 
                           onehot_out=True, 
                           max_args=2, 
                           add_noop=False,
                           use_zero_pad=True,
                           float_labels=True)

def make_dl(ds, ds_test=None):
    train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds_test if ds_test != None else ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
    
    return train_dl, test_dl


def relu(x): return np.maximum(x, 0)


def build_model(w, w_r):
    embedding = {
        0: 0,
        1: 1
    }

    def predict(seq):
        h = 0
        for s in seq:
            h = relu(w * h + embedding[s])
        
        return w_r * h
    
    return predict

def build_model_emb(emb_0, emb_1):
    embedding = {
        0: emb_0,
        1: emb_1
    }

    def predict(seq):
        h = 0
        for s in seq:
            h = relu(2 * h + embedding[s])
        
        return h
    
    return predict

def build_model_2d(w_1, w_2):
    embedding = {
        0: np.array([-1,1]).reshape(-1, 1),
        1: np.array([1,1]).reshape(-1, 1)
    }

    w = np.matrix(f'{w_1},0;,-1,{w_2}')
    readout = np.array([1, -1]).reshape(-1, 1)

    def predict(seq):
        h = np.zeros((2, 1))
        for s in seq:
            h = np.tanh(w @ h + embedding[s])
        
        return readout.T @ h

    return predict

def build_model_3d(w_1, w_2):
    embedding = {
        0: np.array([0,0,0]).reshape(-1, 1),
        1: np.array([1,0,0]).reshape(-1, 1),
        2: np.array([-300, 0, -300]).reshape(-1, 1),
        # 2: np.array([w_1, 0, w_2]).reshape(-1, 1),
    }

    w = np.matrix(f'{w_1},0,0;,{w_2},1,-1;0,0,0')
    readout = np.array([w_2, 1, -1]).reshape(-1, 1)
    # w = np.matrix(f'2,0,0;,1,1,-1;1,0,0')
    # readout = np.array([1, 1, -1]).reshape(-1, 1)

    def predict(seq):
        h = np.zeros((3, 1))
        for s in seq:
            h = relu(w @ h + embedding[s])
        
        return readout.T @ h

    return predict


def get_loss(model, dataset):
    total = 0
    for x, y in dataset:
        pred = model(x.tolist())
        total += (pred - y.item()) ** 2
    
    return total / len(dataset)

def get_loss_manual(model, dataset, w, w_r):
    correct_m = build_model_3d(w, w_r)
    total = 0
    for x, _ in dataset:
        pred = model(x.tolist())
        y = correct_m(x.tolist())
        total += (pred - y) ** 2
    
    return total / len(dataset)

# ds_all = ConcatDataset([ds_args_only, ds_full])
# ds_all = ds_args_only
# ds_all = ds_mini
# ds_all.pad_collate = ds_args_only.pad_collate
# train_dl, test_dl = make_dl(ds_all, None)

# for (x, y), _ in list(zip(ds_all, range(300))):
#     print(x.tolist(), y.item())

# <codecell>
train_dl, test_dl = make_dl(ds_args_only, ds_args_only_test)

model = RnnClassifier1D(fix_emb=True, optim=torch.optim.Adam, full_batch=False).cuda()
traj = [(model.encoder_rnn.weight_hh_l0.item(), model.readout.weight.item())]
def eval_cb(model):
    traj.append((model.encoder_rnn.weight_hh_l0.item(), model.readout.weight.item()))

n_epochs = 1200
eval_every = 10
losses = model.learn(n_epochs, train_dl, test_dl, lr=1e-3, eval_every=eval_every, eval_cb=eval_cb)

# <codecell>
x = np.arange(0, 3, 0.025)
y = np.arange(0, 3, 0.025)

xx, yy = np.meshgrid(x, y)
z = []

for x_, y_ in tqdm(zip(xx.ravel(), yy.ravel()), total=np.prod(xx.shape)):
    m = build_model(x_, y_)
    z.append(get_loss(m, ds_args_only))

z = np.array(z).reshape(xx.shape)

# <codecell>
traj = np.array(traj)

plt.gcf().set_size_inches(4, 3)
plt.annotate('Start', (0.76, 1.5), color='red')

mpb = plt.contourf(xx, yy, np.log(z), 50, vmin=0, extend='both')
plt.axvline(x=2, alpha=0.3, linestyle='dashed', color='black')
plt.axhline(y=1, alpha=0.3, linestyle='dashed', color='black')
plt.colorbar()

plt.plot(traj[:,0], traj[:,1], color='red')
plt.xlabel(r'$w$')
plt.ylabel(r'$w_r$')
plt.savefig('viz/cosyne_fig/rnn_1d_loss_landscape.svg', bbox_inches='tight')

# <codecell>

plt.gcf().set_size_inches(3.5, 2.7)
epochs = np.arange(len(losses['train'])) * eval_every
plt.plot(epochs, losses['train_acc'], label='Train', linestyle='dashed', color='red')
plt.plot(epochs, losses['test_acc'], label='Test', color='red')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.xlim(0, 1100)
plt.legend()

plt.tight_layout()

plt.savefig('viz/cosyne_fig/rnn_1d_acc_plot.svg')

# %%
train_dl, test_dl = make_dl(ds_full, ds_args_only_mini_test)
eval_every = 2

model = RnnClassifier3D(fix_emb=True, optim=torch.optim.Adam, w1=.5, w2=-1.5, full_batch=False).cuda()
traj_1 = [(model.w1.item(), model.w2.item())]
def eval_cb(model):
    traj_1.append((model.w1.item(), model.w2.item()))

n_epochs = 120
losses_1 = model.learn(n_epochs, train_dl, test_dl, lr=1e-3, eval_every=eval_every, eval_cb=eval_cb)

print('new mojo')
model = RnnClassifier3D(fix_emb=True, optim=torch.optim.Adam, w1=.1, w2=-1.5, full_batch=False).cuda()
traj_2 = [(model.w1.item(), model.w2.item())]
def eval_cb2(model):
    traj_2.append((model.w1.item(), model.w2.item()))
losses_2 = model.learn(n_epochs, train_dl, test_dl, lr=1e-3, eval_every=eval_every, eval_cb=eval_cb2)

print('done!')

# %%
x = np.arange(-3, 4, 0.2)
y = np.arange(-2.5, 6, 0.2)

xx, yy = np.meshgrid(x, y)
z = []

for x_, y_ in tqdm(zip(xx.ravel(), yy.ravel()), total=np.prod(xx.shape)):
    m = build_model_3d(x_, y_)
    z.append(get_loss_manual(m, ds_full, 2, 1))

z = np.array(z).reshape(xx.shape)

# <codecell>
plt.gcf().set_size_inches(4, 3)

plt.contourf(xx, yy, np.log(z), 50, vmin=0, extend='both')
plt.axvline(x=2, alpha=0.3, linestyle='dashed', color='black')
plt.axhline(y=1, alpha=0.3, linestyle='dashed', color='black')
plt.colorbar()
plt.annotate('Start', (0.2, -2.1), color='red')


traj_1 = np.array(traj_1)
traj_2 = np.array(traj_2)

plt.plot(traj_1[:,0], traj_1[:,1], color='red')
plt.plot(traj_2[:,0], traj_2[:,1], color='magenta')

plt.xlabel('Blue weight')
plt.ylabel('Red weights (tied)')

plt.savefig('viz/cosyne_fig/rnn_3d_loss_landscape.svg', bbox_inches='tight')

# <codecell>

plt.gcf().set_size_inches(3.5, 2.7)
epochs = np.arange(len(losses_1['train'])) * eval_every
plt.plot(epochs, np.array(losses_1['train_acc']) + 0.005, linestyle='dashed', color='red')
plt.plot(epochs, np.array(losses_1['test_acc']) + 0.005, color='red')

plt.plot(epochs, np.array(losses_2['train_acc']) - 0.01, linestyle='dashed', color='magenta')
plt.plot(epochs, np.array(losses_2['test_acc']) - 0.01, color='magenta')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
# plt.xlim(0, 1100)
plt.legend()

plt.plot((0,0), (0,0), color='k', linestyle='dashed', label='Train')
plt.plot((0,0), (0,0), color='k', label='Test')
plt.legend()

plt.tight_layout()

plt.savefig('cosyne_fig/rnn_3d_acc_plot.svg')