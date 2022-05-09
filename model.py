"""
Model and dataset definitions

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
import json
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset


class BinaryAdditionDataset(Dataset):
    def __init__(self, max_len=None, n_bits=4, little_endian=False, op_filter=None) -> None:
        """
        filter template:

        op_filter = {
            arg1: [(op, [allowed_zero's]), (ditto)],
            arg2: [ditto]
        }
        """
        super().__init__()
        self.end_token = '<END>'
        self.pad_token = '<PAD>'
        self.idx_to_token = ['0', '1', '+', self.end_token, self.pad_token]
        self.token_to_idx = {tok: i for i, tok in enumerate(self.idx_to_token)}
        self.little_endian = little_endian
        self.filter = op_filter

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

                    if self.filter != None and (self._check_match(a_str, 'arg1') or self._check_match(b_str, 'arg2')):
                        continue

                    if self.little_endian:
                        in_str = a_str[::-1] + '+' + b_str[::-1]
                    else:
                        in_str = a_str + '+' + b_str

                    in_toks = [self.end_token] + list(in_str) + [self.end_token]
                    in_toks = [self.token_to_idx[t] for t in in_toks]
                    all_examples.append((in_toks, out_toks))
        
        return all_examples

    # TODO: fails for 0's
    def _check_match(self, tok_str, arg_str):
        arg = int(tok_str, 2)
        for op, n_zeros in self.filter[arg_str]:
            for n in n_zeros:
                if arg == op and tok_str.startswith(n * '0' + '1'):
                    return True

        return False
    
    def args_to_tokens(self, *args):
        answer = np.sum(args)
        if self.little_endian:
            in_str = '+'.join([f'{a:b}'[::-1] for a in args])
        else:
            in_str = '+'.join([f'{a:b}' for a in args])
        in_toks = [self.end_token] + list(in_str) + [self.end_token]

        if self.little_endian:
            out_toks = [self.end_token] + list(f'{answer:b}'[::-1]) + [self.end_token]
        else:
            out_toks = [self.end_token] + list(f'{answer:b}') + [self.end_token]

        in_toks = [self.token_to_idx[t] for t in in_toks]
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
        xs_pad = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ys_pad = pad_sequence(ys, batch_first=True, padding_value=pad_id)
        return xs_pad, ys_pad

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.tensor(x), torch.tensor(y)
    
    def __len__(self):
        return len(self.examples)


class BinaryAdditionLSTM(nn.Module):
    def __init__(self, num_ops=5, end_idx=3, padding_idx=4, embedding_size=5, hidden_size=100, lstm_layers=1) -> None:
        super().__init__()

        self.num_ops = num_ops
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        self.embedding = nn.Embedding(self.num_ops, self.embedding_size)
        self.encoder_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )
        #TODO: is the extra readout needed?
        # self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)
        self.readout = nn.Linear(self.hidden_size, self.num_ops)
    
    def encode(self, input_seq):
        input_lens = torch.sum(input_seq != self.padding_idx, dim=-1)
        input_emb = self.embedding(input_seq)
        input_packed = pack_padded_sequence(input_emb, input_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (enc_h, enc_c) = self.encoder_lstm(input_packed)

        return enc_h, enc_c
    
    def decode(self, input_seq, hidden, cell):
        input_emb = self.embedding(input_seq)
        dec_out, (hidden, cell) = self.decoder_lstm(input_emb, (hidden, cell))
        # preds = self.output_layer(dec_out)
        logits = self.readout(dec_out)
        return logits, hidden, cell

    
    def forward(self, input_seq, output_seq):
        enc_h, enc_c = self.encode(input_seq)

        output_context, output_targets = output_seq[:,:-1], output_seq[:,1:]  #TODO: avoid teacher-forcing 0.5 of the time
        logits, _, _ = self.decode(output_context, enc_h, enc_c)

        mask = output_targets != self.padding_idx
        logits = logits[mask]
        targets = output_targets[mask]
        return logits, targets
    
    def loss(self, logits, targets):
        return nn.functional.cross_entropy(logits, targets)
    
    def trace(self, input_seq, max_steps=100):
        e = self.encoder_lstm
        d = self.decoder_lstm
        sig = torch.sigmoid
        tanh = torch.tanh
        input_seq = torch.tensor(input_seq)

        input_emb = self.embedding(input_seq)
        hidden = torch.zeros((self.hidden_size, 1))
        cell = torch.zeros((self.hidden_size, 1))

        info = {
            'enc': {
                'cell': [],
                'hidden': [],
                'f': [],
                'i': [],
                'o': [],
                'g':[]
            },

            'dec': {
                'cell': [],
                'hidden': [],
                'f': [],
                'i': [],
                'o': [],
                'g':[]
            },
            'input_emb': input_emb,
            'output_emb': [],
            'out': []
        }

        # encode
        for x in input_emb:
            x = x.reshape(-1, 1)
            in_act = e.weight_ih_l0 @ x + e.bias_ih_l0.data.unsqueeze(1)
            hid_act = e.weight_hh_l0 @ hidden + e.bias_hh_l0.data.unsqueeze(1)
            act = in_act + hid_act

            i_gate = sig(act[:self.hidden_size])
            f_gate = sig(act[self.hidden_size:2*self.hidden_size])
            g_writ = tanh(act[2*self.hidden_size:3*self.hidden_size])
            o_gate = sig(act[3*self.hidden_size:])

            cell = f_gate * cell + i_gate * g_writ
            hidden = o_gate * tanh(cell)

            info['enc']['cell'].append(cell)
            info['enc']['hidden'].append(hidden)
            info['enc']['f'].append(f_gate)
            info['enc']['i'].append(i_gate)
            info['enc']['g'].append(g_writ)
            info['enc']['o'].append(o_gate)
        
        # decode
        curr_tok = torch.tensor(self.end_idx)
        gen_out = [curr_tok]
        for _ in range(max_steps):
            x = self.embedding(curr_tok)
            info['output_emb'].append(x)

            x = x.reshape(-1, 1)
            in_act = d.weight_ih_l0 @ x + d.bias_ih_l0.data.unsqueeze(1)
            hid_act = d.weight_hh_l0 @ hidden + d.bias_hh_l0.data.unsqueeze(1)
            act = in_act + hid_act

            i_gate = sig(act[:self.hidden_size])
            f_gate = sig(act[self.hidden_size:2*self.hidden_size])
            g_writ = tanh(act[2*self.hidden_size:3*self.hidden_size])
            o_gate = sig(act[3*self.hidden_size:])

            cell = f_gate * cell + i_gate * g_writ
            hidden = o_gate * tanh(cell)

            # x = self.output_layer.weight @ hidden + self.output_layer.bias.data.unsqueeze(1)
            x = self.readout.weight @ hidden + self.readout.bias.data.unsqueeze(1)
            x = x.flatten()

            curr_tok = torch.argmax(x)
            gen_out.append(curr_tok)

            info['dec']['cell'].append(cell)
            info['dec']['hidden'].append(hidden)
            info['dec']['f'].append(f_gate)
            info['dec']['i'].append(i_gate)
            info['dec']['g'].append(g_writ)
            info['dec']['o'].append(o_gate)

            if curr_tok.item() == self.end_idx:
                break
        
        gen_out = [t.item() for t in gen_out]
        info['out'] = gen_out
        return info
    
    @torch.no_grad()
    def generate(self, input_seq, max_steps=100, device='cpu'):
        input_seq = input_seq.unsqueeze(0)
        h, c = self.encode(input_seq)
        curr_tok = torch.tensor([[self.end_idx]], device=device)
        gen_out = [curr_tok]

        for _ in range(max_steps):
            preds, h, c = self.decode(curr_tok, h, c)
            curr_tok = torch.argmax(preds, dim=-1)
            gen_out.append(curr_tok)
            if curr_tok.item() == self.end_idx:
                break

        return torch.cat(gen_out, dim=-1).squeeze(dim=0)
    
    def save(self, path):
        if type(path) == str:
            path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        torch.save(self.state_dict(), path / 'weights.pt')
        params = {
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'lstm_layers': self.lstm_layers
        }

        with (path / 'params.json').open('w') as fp:
            json.dump(params, fp)
    
    def load(self, path):
        if type(path) == str:
            path = Path(path)

        with (path / 'params.json').open() as fp:
            params = json.load(fp)
        
        self.__init__(**params)
        weights = torch.load(path / 'weights.pt')
        self.load_state_dict(weights)
        self.eval()


class BinaryAdditionRNN(nn.Module):
    def __init__(self, num_ops=5, end_idx=3, padding_idx=4, embedding_size=5, hidden_size=100, n_layers=1) -> None:
        super().__init__()

        self.num_ops = num_ops
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.num_ops, self.embedding_size)
        self.encoder_rnn = nn.RNN(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
        )

        self.decoder_rnn = nn.RNN(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
        )
        #TODO: is the extra readout needed?
        # self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)
        self.readout = nn.Linear(self.hidden_size, self.num_ops)
    
    def encode(self, input_seq):
        input_lens = torch.sum(input_seq != self.padding_idx, dim=-1)
        input_emb = self.embedding(input_seq)
        input_packed = pack_padded_sequence(input_emb, input_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, enc_h = self.encoder_rnn(input_packed)

        return enc_h
    
    def decode(self, input_seq, hidden):
        input_emb = self.embedding(input_seq)
        dec_out, hidden = self.decoder_rnn(input_emb, hidden)
        logits = self.readout(dec_out)
        return logits, hidden

    
    def forward(self, input_seq, output_seq):
        enc_h = self.encode(input_seq)

        output_context, output_targets = output_seq[:,:-1], output_seq[:,1:]
        logits, _ = self.decode(output_context, enc_h)

        mask = output_targets != self.padding_idx
        logits = logits[mask]
        targets = output_targets[mask]
        return logits, targets
    
    def loss(self, logits, targets):
        return nn.functional.cross_entropy(logits, targets)
    
    # TODO: test trace, replace decoder with one-hot <-- STOPPED HERE
    def trace(self, input_seq, max_steps=100):
        e = self.encoder_rnn
        d = self.decoder_rnn
        input_seq = torch.tensor(input_seq)

        input_emb = self.embedding(input_seq)
        hidden = torch.zeros((self.hidden_size, 1))

        info = {
            'enc': {
                'hidden': [],
            },

            'dec': {
                'hidden': [],
            },
            'input_emb': input_emb,
            'output_emb': [],
            'out': []
        }

        # encode
        for x in input_emb:
            x = x.reshape(-1, 1)

            in_act = e.weight_ih_l0 @ x + e.bias_ih_l0.data.unsqueeze(1)
            hid_act = e.weight_hh_l0 @ hidden + e.bias_hh_l0.data.unsqueeze(1)
            hidden = torch.tanh(in_act + hid_act)
            info['enc']['hidden'].append(hidden)
        
        # decode
        curr_tok = torch.tensor(self.end_idx)
        gen_out = [curr_tok]
        for _ in range(max_steps):
            x = self.embedding(curr_tok)
            info['output_emb'].append(x)

            x = x.reshape(-1, 1)
            in_act = d.weight_ih_l0 @ x + d.bias_ih_l0.data.unsqueeze(1)
            hid_act = d.weight_hh_l0 @ hidden + d.bias_hh_l0.data.unsqueeze(1)
            hidden = torch.tanh(in_act + hid_act)

            # x = self.output_layer.weight @ hidden + self.output_layer.bias.data.unsqueeze(1)
            x = self.readout.weight @ hidden + self.readout.bias.data.unsqueeze(1)
            x = x.flatten()

            curr_tok = torch.argmax(x)
            gen_out.append(curr_tok)
            info['dec']['hidden'].append(hidden)

            if curr_tok.item() == self.end_idx:
                break
        
        gen_out = [t.item() for t in gen_out]
        info['out'] = gen_out
        return info
    
    @torch.no_grad()
    def generate(self, input_seq, max_steps=100, device='cpu'):
        input_seq = input_seq.unsqueeze(0)
        h = self.encode(input_seq)
        curr_tok = torch.tensor([[self.end_idx]], device=device)
        gen_out = [curr_tok]

        for _ in range(max_steps):
            preds, h = self.decode(curr_tok, h)
            curr_tok = torch.argmax(preds, dim=-1)
            gen_out.append(curr_tok)
            if curr_tok.item() == self.end_idx:
                break

        return torch.cat(gen_out, dim=-1).squeeze(dim=0)
    
    def save(self, path):
        if type(path) == str:
            path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        torch.save(self.state_dict(), path / 'weights.pt')
        params = {
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers
        }

        with (path / 'params.json').open('w') as fp:
            json.dump(params, fp)
    
    def load(self, path):
        if type(path) == str:
            path = Path(path)

        with (path / 'params.json').open() as fp:
            params = json.load(fp)
        
        self.__init__(**params)
        weights = torch.load(path / 'weights.pt')
        self.load_state_dict(weights)
        self.eval()
    

@torch.no_grad()
def compute_test_loss(model, test_dl):
    all_preds, all_targs = [], []
    for input_seq, output_seq in test_dl:
        input_seq = input_seq.cuda()
        output_seq = output_seq.cuda()

        logits, targets = model(input_seq, output_seq)
        all_preds.append(logits)
        all_targs.append(targets)

    logits = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targs, dim=0)

    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == targets).type(torch.FloatTensor))
    return model.loss(logits, targets).item(), acc.item()
        
@torch.no_grad()
def compute_arithmetic_acc(model, test_dl, ds):
    preds, targets = [], []
    total_correct = 0
    total_correct_no_teacher = 0
    total_count = 0

    for input_batch, output_batch in test_dl:
        for input_seq, output_seq in zip(input_batch, output_batch):
            input_seq = input_seq.unsqueeze(0).cuda()
            output_seq = output_seq.unsqueeze(0).cuda()

            preds_no_teacher = model.generate(input_seq.squeeze(), device='cuda').cpu().numpy()

            logits, targets = model(input_seq, output_seq)
            preds = logits.cpu().numpy().argmax(axis=-1)

            targets = targets.cpu().numpy()

            guess = ds.tokens_to_args(preds)
            guess_no_teacher = ds.tokens_to_args(preds_no_teacher)
            answer = ds.tokens_to_args(targets)

            # print('input_seq', input_seq)
            # print('output_seq', output_seq)
            # print('guess', guess)
            # print('guess no teacher', guess_no_teacher)
            # print('answer', answer)
            # print('total_correct', int(guess == answer))
            total_correct += int(guess == answer)
            total_correct_no_teacher += int(guess_no_teacher == answer)
            total_count += 1
    
    # print('TOTAL COUNT', total_count)
    return total_correct / total_count, total_correct_no_teacher / total_count

'''
# %%
model = BinaryAdditionRNN()
model.load('save/hid5_30k_vargs3_rnn')
print(model)
# %%
info = model.trace([3,1,2,1,1,2,1,3])
print(info['out'])
# %%
'''
