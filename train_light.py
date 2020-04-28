import json
import math
import torch
import torch.nn as nn
from protein_transformer import ProteinTransformer, BindingPredictor2
from models.prop_predictor import PropPredictor
from graph.mol_graph import MolGraph
import multiprocessing
import sys
from utils import model_utils

# lazy, sloppy way of reading in all the data
from get_data import *

args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    torch.set_num_threads(multiprocessing.cpu_count() // 2)

molnet = PropPredictor(args, n_classes = 64)
tmp = PropPredictor(args, n_classes = 1)
state_dict = torch.load('model_87')
tmp.load_state_dict(state_dict)
molnet.model.load_state_dict(tmp.model.state_dict())

for param in molnet.model.parameters():
    param.requires_grad = False

tmp = ProteinTransformer(d_model=64, nhead=4, num_encoder_layers=2,
                         dim_feedforward=32, dropout=0.1, activation='relu')
tmp.load_state_dict(torch.load('pretrained_protein_transformer_2.torch'))

model = BindingPredictor2(tmp.embedder.to(args.device), tmp.encoder.to(args.device))

np.random.seed(42)
batch_inds = np.arange(affinity_tr.shape[0])
np.random.shuffle(batch_inds)
batch_inds_ts = np.arange(affinity_ts.shape[0])
np.random.shuffle(batch_inds_ts)

if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    batch_inds = batch_inds[:3000]
    batch_inds_ts = batch_inds_ts[:1000]

def get_index_batch(i, bsz):
    return batch_inds[i:i+bsz]

def get_mol_descriptors():
    smile_batch, path_batch = combine_data([drug_dataset[0], drug_dataset[1], drug_dataset[2], drug_dataset[3]], args)
    mol_graph = MolGraph(smile_batch, args, path_batch[0], path_batch[1])
    output, _ = molnet.model.forward(mol_graph)
    print(output.size())
    print(mol_graph)
    print(mol_graph.scope)
    outputs = []
    sizes = []
    for j in range(len(drug_dataset.data)):
        smile_batch, path_batch = combine_data([drug_dataset[j]], args)
        path_input, path_mask = path_batch
        path_input = path_input.to(args.device)
        path_mask = path_mask.to(args.device)
        mol_graph = MolGraph(smile_batch, args, path_input, path_mask)
        sizes.append(mol_graph.scope[0][1])
        outputs.append(molnet.model.forward(mol_graph)[0])
    lengths = [output.size(0) for output in outputs]
    print(outputs[0].size(), outputs[1].size(), max(lengths), len(outputs))
    mol_tensor = torch.zeros(len(outputs), max(lengths), outputs[0].size(1), dtype=torch.float32)
    print(mol_tensor.size())
    print(sizes)
    for i, output in enumerate(outputs):
        mol_tensor[i,:lengths[i],:] = output
    return sizes, mol_tensor

sizes, mol_tensor = get_mol_descriptors()
sizes = np.array(sizes, dtype=np.int32)

def setup_batch(i, bsz):
    inds = get_index_batch(i, bsz)
    pr_seq = proteins_tr[inds, :]
    size_set = sizes[drugs_tr_ind[inds]]
    scope_set = [(0,0)]
    for size in size_set:
        scope_set.append((scope_set[-1][0] + scope_set[-1][1], size.item()))
    scope_set = scope_set[1:]
    mol_tensor_set = mol_tensor[drugs_tr_ind[inds]]
    mol_tensor_set = model_utils.convert_to_2D(mol_tensor_set, scope_set)
    return torch.tensor(pr_seq, device = args.device).to(torch.int64), scope_set, mol_tensor_set,\
           torch.tensor(affinity_tr[inds], device=args.device).to(torch.float32)

_, scope_set, mol_set, _ = setup_batch(0, 10)

print(scope_set)
print(mol_set.size())

def setup_batch_ts(i, bsz):
    inds = get_index_batch(i, bsz)
    pr_seq = proteins_tr[inds, :]
    size_set = sizes[drugs_ts_ind[inds]]
    scope_set = [(0, 0)]
    for size in size_set:
        scope_set.append((scope_set[-1][0] + scope_set[-1][1], size.item()))
    scope_set = scope_set[1:]
    mol_tensor_set = mol_tensor[drugs_ts_ind[inds]]
    mol_tensor_set = model_utils.convert_to_2D(mol_tensor_set, scope_set)
    return torch.tensor(pr_seq, device=args.device).to(torch.int64), scope_set, mol_tensor_set, \
           torch.tensor(affinity_tr[inds], device=args.device).to(torch.float32)

criterion = nn.MSELoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

batch_size = 10

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(seq_voc)
    for batch, i in enumerate(range(0, batch_inds.shape[0] - 1, batch_size)):
        pr_seq, scope_set, mol_tensor_set, affinity = setup_batch(i, batch_size)
        optimizer.zero_grad()
        output = model(pr_seq, mol_tensor_set, scope_set)
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
        for i in range(0, batch_inds_ts.shape[0] - 1, batch_size):
            pr_seq, scope_set, mol_tensor_set, affinity = setup_batch_ts(i, batch_size)
            output = eval_model(pr_seq, mol_tensor_set, scope_set)
            total_loss += len(affinity) * criterion(output, affinity).item()
    return total_loss / (len(drugs_ts_ind) - 1)

best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None

model.to(args.device)

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

torch.save(model.state_dict(), 'bapred.th')
