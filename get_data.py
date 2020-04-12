import json
import pickle
from collections import OrderedDict
import numpy as np
from rdkit.Chem import AllChem as Chem
import argparse
from utils.data_utils import load_shortest_paths
from dataset import *

parser = argparse.ArgumentParser()
with open('params.txt', 'r') as f:
    params = f.read()


class Args:
    pass

param_dict = {}
items = params.split()
for item in items:
    key, value = item.split('=')
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    if value == '':
        value = None
    param_dict[key] = value
args = Args()
args.__dict__.update(param_dict)

args.data = 'data/kiba'
load_shortest_paths(args)
drug_dataset = MolDataset(read_smiles_from_file('data/kiba/raw.csv'), args)

# for converting protein sequence to categorical format
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:i for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000   # Note that all protein data will have the same length 1000

def seq_to_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

fpath = 'data/kiba/'

# Read in drugs and proteins
drugs_ = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
drugs = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(d),isomericSmiles=True) for d in drugs_.values()])
proteins_ = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
proteins = np.array(list(proteins_.values()))

# Read in affinity data
affinity = np.array(pickle.load(open(fpath + "Y","rb"), encoding='latin1'))

# Read in train/test fold
train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
train_fold = [ee for e in train_fold for ee in e ]
'''
Here all validation folds are aggregated into training set. 
If you want to train models with different architectures and/or 
optimize for model hyperparameters, we encourage you to use 5-fold 
cross validation as provided here.
'''
test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))

# Prepare train/test data with fold indices
rows, cols = np.where(np.isnan(affinity)==False)
drugs_tr = drugs[rows[train_fold]]
drugs_tr_ind = rows[train_fold]
proteins_tr = np.array([seq_to_cat(p) for p in proteins[cols[train_fold]]])
affinity_tr = affinity[rows[train_fold], cols[train_fold]]

drugs_ts = drugs[rows[test_fold]]
drugs_ts_ind = rows[test_fold]
proteins_ts = np.array([seq_to_cat(p) for p in proteins[cols[test_fold]]])
affinity_ts = affinity[rows[test_fold], cols[test_fold]]
