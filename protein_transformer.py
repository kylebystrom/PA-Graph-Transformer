import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import Transformer, TransformerEncoderLayer, TransformerEncoder

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"


class ProteinEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Simple embedding for a protein sequence which includes a learnable
        encoding for the residues and a fixed positional encoding
        :param d_model: The dimension of the descriptor for each residue.
        :param dropout: Dropout rate
        :param max_len: Length of largest protein sequence in the dataset
        """
        super(ProteinEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.encoder = nn.Embedding(len(seq_voc), d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: Protein sequence as categorical tensor
        :return: tensor shape (l_protein, d_model) embedding and positional encoding
        """
        # print(x.size())
        encod = self.encoder(x)
        # print(encod.size())
        pe = self.pe[:x.size(0), :]
        # print(pe.size())
        x = self.encoder(x) + self.pe[:x.size(0), :]
        return self.dropout(x)


# The following is heavily based on the PyTorch transformer tutorial
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
class ProteinTransformer(nn.Module):

    def __init__(self, d_model=64, nhead=2, num_encoder_layers=4,
                 dim_feedforward=32, dropout=0.1, activation='relu'):
        """
        A Transformer module for a protein sequence.
        :param d_model: Dimension of the attention layers
        :param nhead: number of heads of the attention layer
        :param num_encoder_layers: number of transformer layers in the encoder
        :param dim_feedforward: Dimension of the feedforward layers
        :param dropout: dropout rate
        :param activation: relu or gelu
        """
        super(ProteinTransformer, self).__init__()

        self.embedder = ProteinEmbedding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = nn.Linear(d_model, len(seq_voc))
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a mask for sequence to prevent looking at 'future' residues
        :param sz: length of the protein sequence
        :return: mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        """
        :param x: protein sequence (categorical tensor)
        :return: protein sequence (one-hot tensor)
        """
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask

        x = self.embedder(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BindingPredictor(nn.Module):

    def __init__(self, embedder, encoder, molnet,
                 d_model=64,
                 encoder2_nhead=1, encoder2_dim=32,
                 dropout=0.1, activation='relu'):
        """
        BindingPredictor takes in a protein sequence and MolGraph,
        and outputs a predicted binding affinity between the protein
        sequence and molecule.
        :param embedder: ProteinEmbedding object (possibly pretrained)
        :param encoder: TransformerEncoder object (possibly pretrained)
        :param molnet: PropPredictor object (possible pretrained, with output
            dimension (i.e. n_classes) equal to d_model
        :param d_model: dimension of the embedder, encoder, and molnet
        :param encoder2_nhead: Number of heads in the second encoder
        :param encoder2_dim: Feedforward dimension of the second encoder
        :param dropout: Dropout rate
        :param activation: relu or gelu, used for second encoder
        """
        super(BindingPredictor, self).__init__()

        self.embedder = embedder
        self.encoder = encoder
        self.molnet = molnet

        encoder_layer = TransformerEncoderLayer(d_model, encoder2_dim, encoder2_dim, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder2 = TransformerEncoder(encoder_layer, 1, encoder_norm)

        self.linear1 = nn.Linear(d_model, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, pr_seq, mol_graph):
        # shape (nbatch, lpr, dmodel)
        pr_vec = self.embedder(pr_seq)
        # shape (nbatch, lpr, dmodel)
        pr_enc = self.encoder(pr_vec)

        # shape (nbatch, dmodel)
        atom_feat = self.molnet(mol_graph, None)
        # shape (nbatch, 1, dmodel)
        atom_feat = torch.unsqueeze(atom_feat, 1)

        # shape (nbatch, lpr, dmodel)
        int_vec = pr_enc * atom_feat
        int_enc = self.encoder2(int_vec)

        # shape (nbatch, dmodel)
        int_enc = torch.logsumexp(int_enc, 1)
        # shape (nbatch, 1)
        int_enc = self.linear1(int_enc)
        int_enc = F.relu(int_enc)
        result = self.linear2(int_enc)

        # shape (nbatch)
        return result.squeeze(1)


class BindingPredictor2(nn.Module):

    def __init__(self, embedder, encoder,
                 d_model=64, agg_func = 'sum', molnet_outsize=160,
                 encoder2_nhead=1, encoder2_dim=32,
                 dropout=0.1, activation='relu'):
        """
        BindingPredictor takes in a protein sequence and MolGraph,
        and outputs a predicted binding affinity between the protein
        sequence and molecule.
        :param embedder: ProteinEmbedding object (possibly pretrained)
        :param encoder: TransformerEncoder object (possibly pretrained)
        :param molnet: PropPredictor object (possible pretrained, with output
            dimension (i.e. n_classes) equal to d_model
        :param d_model: dimension of the embedder, encoder, and molnet
        :param encoder2_nhead: Number of heads in the second encoder
        :param encoder2_dim: Feedforward dimension of the second encoder
        :param dropout: Dropout rate
        :param activation: relu or gelu, used for second encoder
        """
        super(BindingPredictor2, self).__init__()

        self.embedder = embedder
        self.encoder = encoder

        encoder_layer = TransformerEncoderLayer(d_model, encoder2_dim, encoder2_dim, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder2 = TransformerEncoder(encoder_layer, 1, encoder_norm)

        # These are for aggregating molecular data.
        self.W_p_h = nn.Linear(molnet_outsize, molnet_outsize)
        self.W_p_o = nn.Linear(molnet_outsize, d_model)
        self.agg_func = agg_func

        self.linear1 = nn.Linear(d_model, 1)
        self.linear2 = nn.Linear(1, 1)

    def aggregate_atom_h(self, atom_h, scope):
        mol_h = []
        for (st, le) in scope:
            cur_atom_h = atom_h.narrow(0, st, le)

            if self.agg_func == 'sum':
                mol_h.append(cur_atom_h.sum(dim=0))
            elif self.agg_func == 'mean':
                mol_h.append(cur_atom_h.mean(dim=0))
            else:
                assert(False)
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h

    def forward(self, pr_seq, atom_h_batch, scope):

        mol_h = self.aggregate_atom_h(atom_h_batch, scope)
        mol_h = nn.ReLU()(self.W_p_h(mol_h))
        # shape (nbatch, dmodel)
        mol_o = self.W_p_o(mol_h)

        # shape (nbatch, lpr, dmodel)
        pr_vec = self.embedder(pr_seq)
        # shape (nbatch, lpr, dmodel)
        pr_enc = self.encoder(pr_vec)

        # shape (nbatch, 1, dmodel)
        atom_feat = torch.unsqueeze(mol_o, 1)

        # shape (nbatch, lpr, dmodel)
        int_vec = pr_enc * atom_feat
        int_enc = self.encoder2(int_vec)

        # shape (nbatch, dmodel)
        int_enc = torch.logsumexp(int_enc, 1)
        # shape (nbatch, 1)
        int_enc = self.linear1(int_enc)
        int_enc = F.relu(int_enc)
        result = self.linear2(int_enc)

        # shape (nbatch)
        return result.squeeze(1)
