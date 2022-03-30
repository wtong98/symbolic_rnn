"""
Training a model to perform binary addition

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

class BinaryAdditionDataset(Dataset):
    def __init__(self, max_len=10000, arg_low=0, arg_high=16) -> None:
        super().__init__()
        self.end_token = '<END>'
        self.pad_token = '<PAD>'
        self.idx_to_token = ['0', '1', '+', self.end_token, self.pad_token]
        self.token_to_idx = {tok: i for i, tok in enumerate(self.idx_to_token)}

        self.examples = [self.args_to_tokens(
            np.random.randint(arg_low, arg_high),
            np.random.randint(arg_low, arg_high)
        ) for _ in range(max_len)]
    
    def args_to_tokens(self, *args):
        answer = np.sum(args)
        in_str = '+'.join([f'{a:b}' for a in args])
        in_toks = [self.end_token] + list(in_str) + [self.end_token]
        out_toks = [self.end_token] + list(f'{answer:b}') + [self.end_token]

        in_toks = [self.token_to_idx[t] for t in in_toks]
        out_toks = [self.token_to_idx[t] for t in out_toks]
        return in_toks, out_toks
    
    def tokens_to_args(self, tokens):
        str_toks = [self.idx_to_token[t] for t in tokens]
        if str_toks[0] == self.end_token:
            str_toks = str_toks[1:]
        if str_toks[-1] == self.end_token:
            str_toks = str_toks[:-1]
        
        str_args = ''.join(str_toks).split('+')
        args = [int(str_a, 2) for str_a in str_args]
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


ds = BinaryAdditionDataset()

test_split = 0.1
test_len = int(len(ds) * test_split)
train_len = len(ds) - test_len

train_ds, test_ds = random_split(ds, [train_len, test_len])

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=32, collate_fn=ds.pad_collate, num_workers=0, pin_memory=True)


# <codecell>
class BinaryAdditionLSTM(nn.Module):
    def __init__(self, num_ops=5, end_idx=3, padding_idx=4) -> None:
        super().__init__()

        self.num_ops = num_ops
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        self.embedding_size = 5
        self.hidden_size = 100
        self.lstm_layers = 1   # TODO: add interpretable tests, try with systematic database <-- STOPPED HERE

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
        self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)
        self.readout = nn.Linear(self.embedding_size, self.num_ops)
    
    def encode(self, input_seq):
        input_lens = torch.sum(input_seq != self.padding_idx, dim=-1)
        input_emb = self.embedding(input_seq)
        input_packed = pack_padded_sequence(input_emb, input_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (enc_h, enc_c) = self.encoder_lstm(input_packed)

        return enc_h, enc_c
    
    def decode(self, input_seq, hidden, cell):
        input_emb = self.embedding(input_seq)
        dec_out, (hidden, cell) = self.decoder_lstm(input_emb, (hidden, cell))
        preds = self.output_layer(dec_out)
        logits = self.readout(preds)
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
    
    @torch.no_grad()
    def generate(self, input_seq, max_steps=100):
        input_seq = input_seq.unsqueeze(0)
        h, c = self.encode(input_seq)
        curr_tok = torch.tensor([[self.end_idx]])
        gen_out = [curr_tok]

        for _ in range(max_steps):
            preds, h, c = self.decode(curr_tok, h, c)
            curr_tok = torch.argmax(preds, dim=-1)
            gen_out.append(curr_tok)
            if curr_tok.item() == self.end_idx:
                break

        return torch.cat(gen_out, dim=-1).squeeze(dim=0)

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
        

# <codecell>
### TRAINING
n_epochs = 35

model = BinaryAdditionLSTM().cuda()

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
model.cpu()

seq = torch.tensor(ds.args_to_tokens(1,2,3)[0])
out = model.generate(seq)
print(ds.tokens_to_args(out))


# <codecell>
model = BinaryAdditionLSTM()

inp = torch.zeros(5, 10, dtype=torch.int32)
out = torch.zeros(5, 10, dtype=torch.int32)

logits, targets = model(inp, out)
# %%
