"""
Model and dataset definitions

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
from ast import Mod
import itertools
import functools
import json
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.utils.data import Dataset


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


class Model(nn.Module):
    def __init__(self, vocab_size=5, end_idx=3, padding_idx=4) -> None:
        super().__init__()
        self.vocab_size=vocab_size
        self.end_idx = end_idx
        self.padding_idx = padding_idx

    def loss(self, logits, targets):
        return nn.functional.cross_entropy(logits, targets)
    
    def save(self, path, params):
        if type(path) == str:
            path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        torch.save(self.state_dict(), path / 'weights.pt')
        with (path / 'params.json').open('w') as fp:
            json.dump(params, fp)
    
    def load(self):
        if type(path) == str:
            path = Path(path)

        with (path / 'params.json').open() as fp:
            params = json.load(fp)
        
        self.__init__(**params)
        weights = torch.load(path / 'weights.pt')
        self.load_state_dict(weights)
        self.eval()
    
    def _train_iter(self, x, y):
        raise NotImplementedError
    
    def evaluate(self, test_dl):
        raise NotImplementedError
    
    def learn(self, n_epochs, train_dl, test_dl, eval_every=100, logging=True, **optim_args):
        self.optimizer = Adam(self.parameters(), **optim_args)

        losses = {'train': [], 'test': [], 'tok_acc': [], 'arith_acc': []}
        running_loss = 0
        running_length = 0

        for e in range(n_epochs):
            for x, y in train_dl:
                x = x.cuda()
                y = y.cuda()
                loss = self._train_iter(x, y)

                running_loss += loss.item()
                running_length += 1

            if (e+1) % eval_every == 0:
                self.eval()
                curr_loss = running_loss / running_length
                test_loss, test_tok_acc, test_arith_acc = self.evaluate(test_dl)

                if logging:
                    print(f'Epoch: {e+1}   train_loss: {curr_loss:.4f}   test_loss: {test_loss:.4f}   tok_acc: {test_tok_acc:.4f}   arith_acc: {test_arith_acc:.4f} ')

                losses['train'].append(curr_loss)
                losses['test'].append(test_loss)
                losses['tok_acc'].append(test_tok_acc)
                losses['arith_acc'].append(test_arith_acc)

                running_loss = 0
                running_length = 0
                self.train()
        
        return losses


# TODO: untested
class Seq2SeqRnnModel(Model):
    def __init__(self, embedding_size=5, hidden_size=100, n_layers=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
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
        self.readout = nn.Linear(self.hidden_size, self.vocab_size)
    
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
        super().save(path, {
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers
        })
    
    def _train_iter(self, x, y):
        self.optimizer.zero_grad()
        logits, targets = self(x, y)
        loss = self.loss(logits, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    @torch.no_grad()
    def evaluate(self, test_dl):
        all_preds, all_targs = [], []
        total_correct = 0
        total_count = 0
        ds = test_dl.dataset

        for input_seq, output_seq in test_dl:
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()

            logits, targets = self(input_seq, output_seq)
            all_preds.append(logits)
            all_targs.append(targets)

            for x, y in zip(input_seq, output_seq):
                preds_no_teacher = self.generate(x, device='cuda').cpu().numpy()
                guess_no_teacher = ds.tokens_to_args(preds_no_teacher)
                answer = ds.tokens_to_args(y)
                total_correct += int(guess_no_teacher == answer)
                total_count += 1

        logits = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targs, dim=0)

        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((preds == targets).type(torch.FloatTensor))

        loss = self.loss(logits, targets).item()
        tok_acc = acc.item()
        arith_acc = total_correct / total_count
        return loss, tok_acc, arith_acc
    
# TODO: untested
class Seq2SeqLstmModel(Seq2SeqRnnModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.encoder_rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )

        self.decoder_rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )

    def encode(self, input_seq):
        input_lens = torch.sum(input_seq != self.padding_idx, dim=-1)
        input_emb = self.embedding(input_seq)
        input_packed = pack_padded_sequence(input_emb, input_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (enc_h, enc_c) = self.encoder_rnn(input_packed)

        return enc_h, enc_c
    
    def decode(self, input_seq, hidden, cell):
        input_emb = self.embedding(input_seq)
        dec_out, (hidden, cell) = self.decoder_rnn(input_emb, (hidden, cell))
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


# TODO: untested
class RnnClassifier(Model):
    def __init__(self, max_arg, embedding_size=5, hidden_size=100, n_layers=1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.max_arg = max_arg
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.encoder_rnn = nn.RNN(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
        )

        self.readout = nn.Linear(self.hidden_size, self.max_arg + 1)

    def encode(self, input_seq):
        input_lens = torch.sum(input_seq != self.padding_idx, dim=-1)
        input_emb = self.embedding(input_seq)
        input_packed = pack_padded_sequence(input_emb, input_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, enc_h = self.encoder_rnn(input_packed)

        return enc_h[-1,...]   # last hidden layer
    
    def forward(self, input_seq):
        enc_h = self.encode(input_seq)
        logits = self.readout(enc_h)
        return logits

    def trace(self, input_seq):
        e = self.encoder_rnn
        input_seq = torch.tensor(input_seq)

        input_emb = self.embedding(input_seq)
        hidden = torch.zeros((self.hidden_size, 1))

        info = {
            'enc': {
                'hidden': [],
            },

            'input_emb': input_emb,
            'logits': None,
            'out': None
        }

        # encode
        for x in input_emb:
            x = x.reshape(-1, 1)

            in_act = e.weight_ih_l0 @ x + e.bias_ih_l0.data.unsqueeze(1)
            hid_act = e.weight_hh_l0 @ hidden + e.bias_hh_l0.data.unsqueeze(1)
            hidden = torch.tanh(in_act + hid_act)
            info['enc']['hidden'].append(hidden)
        
        logits = self.readout(hidden.T).squeeze()
        info['logits'] = logits
        info['out'] = torch.argmax(logits)
        return info

    @torch.no_grad()
    def generate(self, input_seq):
        input_seq = input_seq.unsqueeze(0)
        h = self.encode(input_seq)
        logits = self.readout(h)
        return torch.argmax(logits, dim=-1)
    
    def save(self, path):
        super().save(path, {
            'max_arg': self.max_arg,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers
        })
    
    def _train_iter(self, x, y):
        self.optimizer.zero_grad()
        logits = self(x)
        loss = self.loss(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
    @torch.no_grad()
    def evaluate(self, test_dl):
        all_preds, all_targs = [], []

        for input_seq, targets in test_dl:
            input_seq = input_seq.cuda()
            targets = targets.cuda()

            logits = self(input_seq)
            all_preds.append(logits)
            all_targs.append(targets)

        logits = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targs, dim=0)

        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((preds == targets).type(torch.FloatTensor))
        loss = self.loss(logits, targets).item()
        tok_acc = acc.item()

        return loss, tok_acc, tok_acc


#TODO: untested
class ReservoirClassifier(RnnClassifier):
    def __init__(self, max_arg, n_reservoir_layers=2, activation='linear', **kwargs) -> None:
        super().__init__(max_arg, **kwargs)
        self.n_reservoir_layers = n_reservoir_layers

        self.activ = None
        self.activ_name = activation
        if activation == 'linear':
            self.activ = nn.Identity()
        elif activation == 'tanh':
            self.activ = torch.tanh
        elif activation == 'relu':
            self.activ = torch.relu
        else:
            raise ValueError('unrecognized activation: ', activation)
    
        for i in range(n_reservoir_layers):
            setattr(self, f'reservoir_l{i}', nn.Linear(self.hidden_size, self.hidden_size))

        # freeze encoder rnn
        for param in self.encoder_rnn.parameters():
            param.requires_grad = False

    def forward(self, input_seq):
        enc_h = self.encode(input_seq)

        for i in range(self.n_reservoir_layers):
            layer = getattr(self, f'reservoir_l{i}')
            enc_h = self.activ(layer(enc_h))
            enc_h = layer(enc_h)

        logits = self.readout(enc_h)
        return logits

    def trace(self, input_seq):
        e = self.encoder_rnn
        input_seq = torch.tensor(input_seq)

        input_emb = self.embedding(input_seq)
        hidden = torch.zeros((self.hidden_size, 1))

        info = {
            'enc': {
                'hidden': [],
            },

            'input_emb': input_emb,
            'logits': None,
            'out': None
        }

        # encode
        for x in input_emb:
            x = x.reshape(-1, 1)

            in_act = e.weight_ih_l0 @ x + e.bias_ih_l0.data.unsqueeze(1)
            hid_act = e.weight_hh_l0 @ hidden + e.bias_hh_l0.data.unsqueeze(1)
            hidden = torch.tanh(in_act + hid_act)
            info['enc']['hidden'].append(hidden)
        
        hidden = hidden.T
        for i in range(self.n_reservoir_layers):
            layer = getattr(self, f'reservoir_l{i}')
            hidden = self.activ(layer(hidden))

        logits = self.readout(hidden).squeeze()
        info['logits'] = logits
        info['out'] = torch.argmax(logits)
        return info

    def save(self, path):
        Model.save(self, path, {
            'max_arg': self.max_arg,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'n_reservoir_layers': self.n_reservoir_layers,
            'activation': self.activ_name
        })


# TODO: make more efficient
class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):
        seqs, lens = pad_packed_sequence(input, batch_first=True)

        all_hidden = []
        for s, l in zip(seqs, lens):
            hidden = torch.zeros(1, self.hidden_size).cuda()
            for idx in range(l):
                hidden = self.ih(s[idx,:]) + self.hh(hidden)
            
            all_hidden.append(hidden)

        return None, torch.concat(all_hidden, dim=0)


# TODO: untested
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

'''
# %%
model = BinaryAdditionFlatRNN(0)
model.load('save/hid5_50k_vargs3_rnn_flat')
print(model)
# %%
info = model.trace([3,1,2,1,3])
print(info['out'])
# %%
'''
