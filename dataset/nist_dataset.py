import os
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import lmdb
import pickle
import numpy as np
import torch
from scipy.spatial import distance_matrix
from functools import lru_cache
from .dactionary import Dictionary
import periodictable
from .collate_fn import Multi_process_batch_collate_fn
from collections import defaultdict
from rdkit import Chem




class Multi_process_NISTDataset(Dataset):
    def __init__(self, db_path,dict_path,config):
        # 初始化函数，传入数据库路径、字典路径和配置
        self.db_path = db_path
        # 断言数据库路径是否存在
        self.config = config
        # 连接数据库
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        # 获取数据库中的所有键
        env = self.connect_db(self.db_path)
        with env.begin() as txn: # type: ignore
            self._keys = list(txn.cursor().iternext(values=False))
        self.dictionary = Dictionary.load(dict_path)

    
    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def get_unimol_data(self, atoms, smi, ir, coordinates, dictionary, max_atoms=512, remove_Hs=False):
        atoms = np.array(atoms)
        assert len(atoms) == len(coordinates) and len(atoms) > 0
        assert coordinates.shape[1] == 3

        ## Remove Hydrogen atoms
        if remove_Hs:
            mask_hydrogen = atoms != "H"
            if sum(mask_hydrogen) > 0:
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]

        ## Randomly sample atoms if the number of atoms is larger than max_atoms
        if len(atoms) > max_atoms:
            index = np.random.permutation(len(atoms))[:max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        assert 0 < len(atoms) <= max_atoms

        atom_vec = torch.from_numpy(dictionary.vec_index(atoms)).long()
        atom_vec = torch.cat([torch.LongTensor([dictionary.bos()]), 
                                atom_vec, 
                                torch.LongTensor([dictionary.eos()])])

        coordinates = coordinates - coordinates.mean(axis=0)
        coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)

        return {'src_tokens': atom_vec, 'src_edge_type': edge_type, 'src_coord':coordinates, 'src_distance': dist, 'smi': smi, 'ir': ir}

    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        try:
            data = pickle.loads(datapoint_pickled)
        except:
            print(idx)
        smi = data["smi"]
        ir = data["ir"]
        ir = ir.unsqueeze(0).unsqueeze(1)
        atoms = data["atoms"]
        atoms = [periodictable.elements[int(z)].symbol for z in atoms]
        coordinates = data["coordinates"]
        data_graph = defaultdict(list)
        data_graph['unimol'].append(self.get_unimol_data(atoms, smi, ir, coordinates, self.dictionary))
        edge_index = data["edge_index"]
        atom_feat = data["atom_attr"]
        edge_feat = data["edge_attr"]
        atomic_number = data["atoms"]
        pos = data["coordinates"]
        data_graph['pyg'].append(Data(x=atom_feat, atomic_number=atomic_number,pos=pos,
                                        edge_index=edge_index, 
                                        edge_attr=edge_feat))
        return data_graph
    
    def __len__(self):
        return len(self._keys)