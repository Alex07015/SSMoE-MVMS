from dataclasses import dataclass


@dataclass
class MolConfig:
    mol_pretrain_pth: str = "/home/zjh/project/Mol_GPT/ckpt/mol_pre_all_h_220816.pt"
    mol_dict_pth: str = "/home/zjh/project/Mol_GPT/data/dict.txt"