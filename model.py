"""
Model and dataset definitions

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
import itertools
import functools
import json
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split


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


class Model(nn.Module):
    def __init__(self, loss_func='bce', vocab_size=6, end_idx=3, padding_idx=4, ewc_weight=0, l1_weight=0, optim=None, full_batch=False) -> None:
        super().__init__()
        self.loss_func = loss_func
        self.vocab_size=vocab_size
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        self.l1_weight = l1_weight
        self.full_batch = full_batch

        self.optim = optim
        if self.optim == None:
            self.optim = Adam

        self.ewc_weight = ewc_weight
        self.use_ewc = False
        self.old_params = None
        self.fisher_info = None

    def loss(self, logits, targets):
        ewc_loss = 0
        l1_loss = 0
        loss = 0

        if self.use_ewc:
            curr_params = torch.concat([p.flatten() for p in self.parameters()])
            ewc_loss = torch.sum(self.fisher_info * (curr_params - self.old_params) ** 2)
        
        if self.l1_weight > 0:
            params = self.collect_reg_params()
            l1_loss = torch.norm(params, p=1) / len(params)

        if self.loss_func == 'bce':
            loss = nn.functional.cross_entropy(logits, targets) 
        else:
            loss = nn.functional.mse_loss(logits, targets.unsqueeze(1))
        
        return loss + self.ewc_weight * ewc_loss + self.l1_weight * l1_loss
    
    def save(self, path, params):
        if type(path) == str:
            path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        torch.save(self.state_dict(), path / 'weights.pt')
        with (path / 'params.json').open('w') as fp:
            json.dump(params, fp)
    
    def load(self, path, device='cpu'):
        if type(path) == str:
            path = Path(path)

        with (path / 'params.json').open() as fp:
            params = json.load(fp)
        
        self.__init__(**params)
        weights = torch.load(path / 'weights.pt', device)
        self.load_state_dict(weights)
        self.eval()
    
    def _train_iter(self, x, y):
        raise NotImplementedError
    
    def evaluate(self, test_dl):
        raise NotImplementedError
    
    def collect_reg_params(self):
        raise NotImplementedError
    
    def fix_ewc(self, train_dl):
        self.old_params = []
        self.fisher_info = []

        for p in self.parameters():
            p.grad.zero_()

        for x, y in train_dl:
            x = x.cuda()
            y = y.cuda()

            # TODO: only works with *Classifier models
            logits = self(x)
            loss = self.loss(logits, y)
            loss.backward()
            

        for p in self.parameters():
            self.old_params.append(p.data.detach().flatten())
            grad = p.grad.detach().flatten()
            self.fisher_info.append(grad ** 2 / len(train_dl))
        
        self.old_params = torch.concat(self.old_params)
        self.fisher_info = torch.concat(self.fisher_info)
        self.use_ewc = True
    
    def learn(self, n_epochs, train_dl, test_dl, eval_every=100, logging=True, eval_cb=None, **optim_args):
        self.optimizer = self.optim(self.parameters(), **optim_args)
        is_cuda = False
        if next(self.parameters()).device != torch.device('cpu'):
            is_cuda = True

        losses = {'train': [], 'test': [], 'tok_acc': [], 'arith_acc': []}
        running_loss = 0
        running_length = 0

        for e in range(n_epochs):
            self.optimizer.zero_grad()

            for x, y in train_dl:
                if is_cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss = self._train_iter(x, y)

                if not self.full_batch:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                running_loss += loss.item()
                running_length += 1
            
            if self.full_batch:
                self.optimizer.step()

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

                if eval_cb != None:
                    eval_cb(self)
        
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
        logits, targets = self(x, y)
        loss = self.loss(logits, targets)
        loss.backward()
        return loss
    
    @torch.no_grad()
    def evaluate(self, test_dl):
        all_preds, all_targs = [], []
        total_correct = 0
        total_count = 0
        ds = test_dl.dataset
        is_cuda = next(self.parameters).device != torch.device('cpu')

        for input_seq, output_seq in test_dl:
            if is_cuda:
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


class RnnClassifier(Model):
    def __init__(self, max_arg, nonlinearity='tanh', use_softexp=False,
                 embedding_size=5, hidden_size=100, n_layers=1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.max_arg = max_arg
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.use_softexp = use_softexp

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.encoder_rnn = nn.RNN(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            nonlinearity=nonlinearity
        )

        if self.use_softexp:
            self.hidden = nn.Linear(self.hidden_size, 2 * self.hidden_size)

        if self.loss_func == 'bce':
            self.readout = nn.Linear(self.hidden_size, self.max_arg + 1)  
        elif self.loss_func == 'mse':
            self.readout = nn.Linear(self.hidden_size, 1)
        else:
            raise ValueError('loss_func should be either "bce" or "mse"')

    def encode(self, input_seq):
        input_lens = torch.sum(input_seq != self.padding_idx, dim=-1)
        input_emb = self.embedding(input_seq)
        input_packed = pack_padded_sequence(input_emb, input_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, enc_h = self.encoder_rnn(input_packed)

        if type(enc_h) == tuple:
            enc_h = enc_h[0]

        return enc_h[-1,...]   # last hidden layer
    
    def forward(self, input_seq):
        hid = self.encode(input_seq)

        if self.use_softexp:
            hid = self.hidden(hid)
            alpha, hid = torch.split(hid, [self.hidden_size, self.hidden_size], dim=1)
            alpha = torch.sigmoid(alpha)

            # final_hid = torch.zeros(alpha.shape, device=next(self.parameters()).device)
            # final_hid[alpha>0] = ((2 ** (alpha * hid) - 1) / alpha + alpha)[alpha>0]
            # final_hid[alpha==0] = hid[alpha==0]
            # final_hid[alpha<0] = (-torch.log2(1 - alpha * (hid + alpha))/alpha)[alpha<0]

            # hid = (2 ** (alpha * hid) - 1) / alpha + alpha
            hid = alpha * 2 ** hid + (1 - alpha) * hid   # swap to linear interpolation
            # hid = 2 ** hid

        logits = self.readout(hid)
        return logits

    def trace(self, input_seq):
        e = self.encoder_rnn
        input_seq = torch.tensor(input_seq)
        activ_f = torch.tanh if self.nonlinearity == 'tanh' else torch.relu

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
            hidden = activ_f(in_act + hid_act)
            info['enc']['hidden'].append(hidden)
        
        if self.use_softexp:
            hid = self.hidden(hidden.T)
            alpha, hid = torch.split(hid, [self.hidden_size, self.hidden_size], dim=1)
            alpha = torch.sigmoid(alpha)
            hidden = (2 ** (alpha * hid) - 1) / alpha + alpha
            hidden = hidden.T
            
        logits = self.readout(hidden.T).squeeze()
        info['logits'] = logits
        info['out'] = torch.argmax(logits)
        return info

    @torch.no_grad()
    def generate(self, input_seq):
        input_seq = input_seq.unsqueeze(0)
        logits = self(input_seq)  # TODO: untested
        return torch.argmax(logits, dim=-1)
    
    @torch.no_grad()
    def get_embedding(self, token_idxs):
        embs = self.embedding(torch.tensor(token_idxs))
        embs = self.encoder_rnn.weight_ih_l0 @ embs.T \
            + self.encoder_rnn.bias_ih_l0.unsqueeze(1) \
            + self.encoder_rnn.bias_hh_l0.unsqueeze(1)
        return embs
    
    def save(self, path):
        super().save(path, {
            'max_arg': self.max_arg,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'use_softexp': self.use_softexp,
            'nonlinearity': self.nonlinearity
        })
    
    def _train_iter(self, x, y):
        logits = self(x)
        loss = self.loss(logits, y)
        loss.backward()
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
    
    def collect_reg_params(self):
        params = [
            self.embedding.weight,
            *self.encoder_rnn.parameters(),
            self.readout.weight
        ]
        params = torch.cat([p.view(-1) for p in params])
        return params


class RnnClassifierWithMLP(RnnClassifier):
    def __init__(self, n_mlp_layers=2, activation='tanh', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_mlp_layers = n_mlp_layers

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
    
        for i in range(n_mlp_layers):
            setattr(self, f'mlp_l{i}', nn.Linear(self.hidden_size, self.hidden_size))
        

    def forward(self, input_seq):
        enc_h = self.encode(input_seq)

        for i in range(self.n_mlp_layers):
            layer = getattr(self, f'mlp_l{i}')
            enc_h = self.activ(layer(enc_h))

        logits = self.readout(enc_h)
        return logits


    def save(self, path):
        Model.save(self, path, {
            'max_arg': self.max_arg,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'n_mlp_layers': self.n_mlp_layers,
            'activation': self.activ_name
        })

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
        for i in range(self.n_mlp_layers):
            layer = getattr(self, f'mlp_l{i}')
            hidden = self.activ(layer(hidden))

        logits = self.readout(hidden).squeeze()
        info['logits'] = logits
        info['out'] = torch.argmax(logits)
        return info


class ReservoirClassifier(RnnClassifier):
    def __init__(self, max_arg, n_reservoir_layers=2, activation='tanh', **kwargs) -> None:
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
            hidden_chunk = torch.tanh(self.ih(b) + self.hh(hidden[:size,...]))
            hidden = torch.cat((hidden_chunk, hidden[size:,...]), dim=0)

        hidden = hidden[unsort_idxs]
        return None, hidden.unsqueeze(0)

class LinearRNNwithSoftExp(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, 2 * hidden_size)

    def forward(self, input_pack):
        dev = next(self.parameters()).device

        data, batch_sizes, _, unsort_idxs = input_pack
        batch_idxs = batch_sizes.cumsum(0)
        batches = torch.tensor_split(data, batch_idxs[:-1])

        hidden = torch.zeros(batch_sizes[0], self.hidden_size).to(dev)
        for b, size in zip(batches, batch_sizes):
            hid = self.hh(hidden[:size,...])

            alpha, hid = torch.split(hid, [self.hidden_size, self.hidden_size], dim=1)
            alpha = torch.sigmoid(alpha)
            hid = (2 ** (alpha * hid) - 1) / alpha + alpha

            hidden_chunk = torch.tanh(self.ih(b) + hid)
            hidden = torch.cat((hidden_chunk, hidden[size:,...]), dim=0)

        hidden = hidden[unsort_idxs]
        return None, hidden.unsqueeze(0)



class LinearRnnClassifier(RnnClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder_rnn = LinearRNNwithSoftExp(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
        )
    
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

            in_act = e.ih.weight @ x + e.ih.bias.data.unsqueeze(1)
            hid_act = e.hh.weight @ hidden + e.hh.bias.data.unsqueeze(1)
            hidden = in_act + hid_act
            info['enc']['hidden'].append(hidden)
        
        logits = self.readout(hidden.T).squeeze()
        info['logits'] = logits
        info['out'] = torch.argmax(logits)
        return info


class LstmClassifier(RnnClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder_rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
        )

    def trace(self, input_seq):
        e = self.encoder_rnn
        sig = torch.sigmoid
        tanh = torch.relu
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

            'input_emb': input_emb,
            'logits': None,
            'out': None
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
        
        logits = self.readout(hidden.T).squeeze()
        info['logits'] = logits
        info['out'] = torch.argmax(logits)
        return info


class NtmClassifier(RnnClassifier):
    def __init__(self, max_arg, embedding_size=5, ctrl_size=100, mem_size=100, word_size=8, **kwargs) -> None:
        super().__init__(max_arg, embedding_size, hidden_size=ctrl_size, **kwargs)

        self.ctrl_size = ctrl_size
        self.mem_size = mem_size
        self.word_size = word_size

        self.encoder_rnn = None
        self.ntm = Ntm(ctrl_size=ctrl_size, mem_size=mem_size, word_size=word_size)
        self.readout = nn.Linear(ctrl_size + word_size, self.max_arg + 1)
    
    def encode(self, input_seq):
        input_lens = torch.sum(input_seq != self.padding_idx, dim=-1)
        input_emb = self.embedding(input_seq)
        input_packed = pack_padded_sequence(input_emb, input_lens.cpu(), batch_first=True, enforce_sorted=False)
        state = self.ntm(input_packed)

        return torch.cat((state['ctrl_hid'], state['read']), dim=1)
    
    def trace(self, input_seq):
        raise NotImplementedError
    
    def get_embedding(self, token_idxs):
        raise NotImplementedError
    
    def save(self, path):
        Model.save(self, path, {
            'max_arg': self.max_arg,
            'embedding_size': self.embedding_size,
            'ctrl_size': self.ctrl_size,
            'mem_size': self.mem_size,
            'word_size': self.word_size,
        })


class Ntm(nn.Module):
    def __init__(self, ctrl_size=32, n_read_heads=1, n_write_heads=1, mem_size=64, word_size=32) -> None:
        super().__init__()
        
        self.controller = NtmLstmController(ctrl_size=ctrl_size, word_size=word_size)
        self.memory = NtmMemory(ctrl_size=ctrl_size, n_read_heads=n_read_heads, n_write_heads=n_write_heads, mem_size=mem_size, word_size=word_size)
    
    def forward(self, input_pack):
        dev = next(self.parameters()).device

        data, batch_sizes, _, unsort_idxs = input_pack
        batch_idxs = batch_sizes.cumsum(0)
        batches = torch.tensor_split(data, batch_idxs[:-1])

        ctrl_state = self.controller.init_state(batch_sizes[0], dev)
        mem_state = self.memory.init_state(batch_sizes[0], dev)
        state = dict(**ctrl_state, **mem_state)

        for b, size in zip(batches, batch_sizes):
            old_state, state = state, self._idx_state(state, size)

            ctrl_in = torch.cat((b, state['read']), dim=1)
            state = self.controller(ctrl_in, state)
            state = self.memory(state)

            state = self._merge_state(state, old_state, size)
        
        return self._sort_state(state, unsort_idxs)
    
    def _idx_state(self, state, size):
        return {k: v[:size,] for k, v in state.items()}
    
    def _merge_state(self, upd_state, old_state, size):
        return {k: torch.cat((upd_state[k], old_state[k][size:,]), dim=0) for k in upd_state}
    
    def _sort_state(self, state, idxs):
        return {k: v[idxs] for k, v in state.items()}


class NtmLstmController(nn.Module):
    def __init__(self, embedding_size=5, ctrl_size=32, word_size=32) -> None:
        super().__init__()

        self.ctrl_size = ctrl_size
        self.embedding_size = embedding_size
        self.word_size = word_size

        self.model = nn.LSTM(
            input_size=self.embedding_size + self.word_size,
            hidden_size=self.ctrl_size,
            num_layers=1,   #TODO: consider multiple layers?
            batch_first=True,
        )
        
    def forward(self, ctrl_in, state):
        _, (new_hid, new_cell) = self.model(ctrl_in.unsqueeze(1), (state['ctrl_hid'].unsqueeze(0), state['ctrl_cell'].unsqueeze(0)))
        state['ctrl_hid'] = new_hid.squeeze(0)
        state['ctrl_cell'] = new_cell.squeeze(0)
        return state
    
    def init_state(self, batch_size, device='cpu'):
        return {
            'read': torch.zeros(batch_size, self.word_size, device=device),
            'ctrl_hid': torch.zeros(batch_size, self.ctrl_size, device=device),
            'ctrl_cell': torch.zeros(batch_size, self.ctrl_size, device=device)
        }


class NtmMemory(nn.Module):
    def __init__(self, ctrl_size=32, n_read_heads=1, n_write_heads=1, mem_size=64, word_size=32) -> None:
        super().__init__()

        self.ctrl_size = ctrl_size
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.mem_size = mem_size
        self.word_size = word_size

        self.fc_read = nn.Linear(ctrl_size, self.word_size + 6)   # key and address
        self.fc_write = nn.Linear(ctrl_size, 3 * self.word_size + 6)   # key, erase, add, and address

        self.register_buffer('mem_init', torch.Tensor(self.mem_size, self.word_size))
        sd = 1 / np.sqrt(self.mem_size + self.word_size)
        nn.init.uniform_(self.mem_init, -sd, sd)
    
    def init_state(self, batch_size, device='cpu'):
        return {
            'mem': self.mem_init.clone().repeat(batch_size, 1, 1),
            'prev_w_write': torch.zeros(batch_size, self.mem_size, device=device),
            'prev_w_read': torch.zeros(batch_size, self.mem_size, device=device)
        }
    
    def forward(self, state):
        write_info = self.fc_write(state['ctrl_hid'])
        state = self.write(state, write_info)

        read_info = self.fc_read(state['ctrl_hid'])
        state = self.read(state, read_info)
        return state
    
    def write(self, state, write_info):
        mem = state['mem']
        prev_w = state['prev_w_write']

        key, erase, add, beta, g, s, gamma = torch.split(write_info, 3 * [self.word_size] + [1, 1, 3, 1], dim=1)
        w = self.address(mem, key, beta, g, s, gamma, prev_w)
        state['mem'] = mem * (1 - w.unsqueeze(-1) @ erase.unsqueeze(1)) + w.unsqueeze(-1) @ add.unsqueeze(1)
        return state

    def read(self, state, read_info):
        mem = state['mem']
        prev_w = state['prev_w_read']

        key, beta, g, s, gamma = torch.split(read_info, [self.word_size, 1, 1, 3, 1], dim=1)
        w = self.address(mem, key, beta, g, s, gamma, prev_w)
        state['read'] = (w.unsqueeze(1) @ mem).squeeze(1)
        return state
    
    def address(self, mem, key, beta, g, s, gamma, prev_w):
        # attention (content-based addressing)
        # TODO: try scaled dot-product attention
        key = torch.tanh(key)
        beta = torch.relu(beta)
        w = torch.softmax(beta * F.cosine_similarity(mem, key.unsqueeze(1), dim=-1), dim=1)

        # interpolate
        g = torch.sigmoid(g)
        w = g * w + (1 - g) * prev_w

        # shift
        s = F.softmax(s, dim=1)
        w = torch.stack([self._conv(w[b], s[b]) for b in range(w.shape[0])])

        # sharpen
        gamma = 1 + torch.relu(gamma)
        w = w ** gamma / torch.sum(w ** gamma)
        return w
    
    def _conv(self, w, s):
        w = torch.cat((w[-1:], w, w[:1]))
        w = F.conv1d(w.reshape(1, 1, -1), s.reshape(1, 1, -1)).reshape(-1)
        return w

'''

# <codecell>
# TODO: try without using fixed max args
ds = BinaryAdditionDataset(n_bits=2, 
                           onehot_out=True, 
                           max_args=3, 
                           add_noop=True,
                           max_noop=5,
                        #    max_noop_only=True,
                        #    max_only=True, 
                           little_endian=False,
                           filter_={
                               'in_args': []
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
model = NtmClassifier(
    max_arg=9,
    embedding_size=5,
    ctrl_size=100,
    mem_size=100,
    word_size=8,
    vocab_size=6).cuda()

# model.load('save/hid100k_vargs3_nbits3_linear')

# <codecell>
### TRAINING
n_epochs = 100
losses = model.learn(n_epochs, train_dl, test_dl, lr=1e-4, eval_every=5)

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
            

# %%
'''