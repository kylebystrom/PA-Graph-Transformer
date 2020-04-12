import json
import math
import torch
import torch.nn as nn
from protein_transformer import ProteinTransformer, BindingPredictor
from models.prop_predictor import PropPredictor
from graph.mol_graph import MolGraph

# lazy, sloppy way of reading in all the data
from get_data import *

molnet = PropPredictor(args, n_classes = 64)
tmp = PropPredictor(args, n_classes = 1)
state_dict = torch.load('model_87')
tmp.load_state_dict(state_dict)
molnet.model.load_state_dict(tmp.model.state_dict())

tmp = ProteinTransformer(d_model=64, nhead=4, num_encoder_layers=2,
                         dim_feedforward=32, dropout=0.1, activation='relu')
tmp.load_state_dict(torch.load('pretrained_protein_transformer_2.torch'))

model = BindingPredictor(tmp.embedder, tmp.encoder, molnet)

np.random.seed(42)
batch_inds = np.arange(affinity_tr.shape[0])
np.random.shuffle(batch_inds)
batch_inds_ts = np.arange(affinity_ts.shape[0])
np.random.shuffle(batch_inds_ts)

def get_index_batch(i, bsz):
    return batch_inds[i:i+bsz]

def setup_batch(i, bsz):
    inds = get_index_batch(i, bsz)
    pr_seq = proteins_tr[inds, :]
    smile_batch, path_batch = combine_data([drug_dataset[j] for j in drugs_tr_ind[inds]], args)
    path_input, path_mask = path_batch
    mol_graph = MolGraph(smile_batch, args, path_input, path_mask)
    return torch.tensor(pr_seq).to(torch.int64), mol_graph, torch.tensor(affinity_tr[inds]).to(torch.float32)

def setup_batch_ts(i, bsz):
    inds = batch_inds_ts[i:i+bsz]
    pr_seq = proteins_tr[inds, :]
    smile_batch, path_batch = combine_data([drug_dataset[j] for j in drugs_ts_ind[inds]], args)
    path_input, path_mask = path_batch
    mol_graph = MolGraph(smile_batch, args, path_input, path_mask)
    return torch.tensor(pr_seq).to(torch.int64), mol_graph, torch.tensor(affinity_ts[inds]).to(torch.float32)

criterion = nn.MSELoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

batch_size = 100

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(seq_voc)
    for batch, i in enumerate(range(0, drugs_tr_ind.shape[0] - 1, batch_size)):
        pr_seq, mol_graph, affinity = setup_batch(i, batch_size)
        optimizer.zero_grad()
        output = model(pr_seq, mol_graph)
        loss = criterion(output, affinity)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(affinity) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(seq_voc)
    count = 0
    with torch.no_grad():
        for i in range(0, drugs_ts_ind.size(0) - 1, batch_size):
            pr_seq, mol_graph, affinity = setup_batch_ts(i, batch_size)
            output = eval_model(pr_seq, mol_graph)
            total_loss += len(affinity) * criterion(output, affinity).item()
    return total_loss / (len(drugs_ts_ind) - 1)

best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
