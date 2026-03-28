'''
Adopted from OGB
* https://github.com/snap-stanford/ogb/blob/1.3.2/ogb/graphproppred/mol_encoder.py
'''

import torch
from .ogb_features import get_atom_feature_dims, get_bond_feature_dims # Modified line

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            # feature = x[:, i]
            # # ⚠️ 先转到 CPU 再做 max/min，避免 CUDA 异步报错
            # feature_cpu = feature.detach().cpu()

            # max_val = feature_cpu.max().item()
            # min_val = feature_cpu.min().item()

            # print(f"Feature {i}: min={min_val}, max={max_val}")
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   