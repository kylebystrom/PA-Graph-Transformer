from datasets.mol_dataset import MolDataset
import torch.utils.data
import rdkit.Chem as Chem
import pdb
from utils.data_utils import load_shortest_paths
from utils import path_utils


def read_smiles_from_file(data_path):
    smiles_data = []

    data_file = open(data_path, 'r')
    for line in data_file.readlines():
        smiles = line.strip().split(',')[0]
        smiles_data.append(smiles)
    data_file.close()
    return smiles_data


class MolDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, args):
        self.args = args

        data = raw_data
        self.data = data

    def __getitem__(self, index):
        smiles = self.data[index]
        mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()

        path_input = None
        path_mask = None
        self.args.use_paths = True
        if self.args.use_paths:
            shortest_paths = [self.args.p_info[smiles]]
            path_input, path_mask = path_utils.get_path_input(
                [mol], shortest_paths, n_atoms, self.args, output_tensor=False)
            path_input = path_input.squeeze(0)  # Remove batch dimension
            path_mask = path_mask.squeeze(0)  # Remove batch dimension
        return smiles, n_atoms, (path_input, path_mask)

    def __len__(self):
        return len(self.data)


def combine_data(data):
    batch_smiles, batch_n_atoms, batch_path = zip(*data)

    batch_path_inputs, batch_path_masks = zip(*batch_path)

    if batch_path_inputs[0] is not None:  # This means paths are used
        # print('paths!')
        max_atoms = max(batch_n_atoms)
        batch_path_inputs, batch_path_mask = path_utils.merge_path_inputs(
            batch_path_inputs, batch_path_masks, max_atoms, args)
        print(batch_path_inputs[0].shape)
    else:
        batch_path_inputs = None
        batch_path_mask = None
    return batch_smiles, (batch_path_inputs, batch_path_mask)


#args.data = 'data/kiba'
#load_shortest_paths(args)
#drug_dataset = MolDataset(read_smiles_from_file('data/kiba/raw.csv'), args)
# args.p_info['COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)Cl)Cl)Cl']