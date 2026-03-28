import logging
import random
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_dir, ".."))
from model.ssmoe_mvms import SSMoE_MVMS
from model.ssmoe_mvms_retrieval import SSMoE_MVMS_Retrieval
from model.model_config import MolConfig
from dataset.qm9s_dataset import Multi_process_Qm9sDataset
from dataset.nist_dataset import Multi_process_NISTDataset
from dataset.collate_fn import Multi_process_batch_collate_fn# type: ignore
import math
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


'''******* 初始化相关  *******'''
def init_seed(manual_seed):
    """
    Set random seed for torch and numpy.
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filepath, log_title):
    # 使用 filepath 作为 logger 的名字
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)

    # ✅ 只添加一次 handler（防止重复）
    if not logger.handlers:
        # 创建目录（如果需要）
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # 文件输出
        fh = logging.FileHandler(filepath)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # 可选：控制台输出（如不需要可以注释掉）
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        # 起始分隔符
        logger.info('-' * 54 + log_title + '-' * 54)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)


# 预训练的dataloader
def get_dataloaders(config,args):
    print("---------------Loading the dataset-------------------")
    if args.dataset == "qm9s_spectra":
        train_dataset = Multi_process_Qm9sDataset(config["train_db_path"],config["dict_path"],config)
        val_dataset = Multi_process_Qm9sDataset(config["val_db_path"],config["dict_path"],config)
        test_dataset = Multi_process_Qm9sDataset(config["test_db_path"],config["dict_path"],config)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True,pin_memory=True,collate_fn=Multi_process_batch_collate_fn)
        val_data_loader = torch.utils.data.DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=True,pin_memory=True,collate_fn=Multi_process_batch_collate_fn)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=True,pin_memory=True,collate_fn=Multi_process_batch_collate_fn)

    elif args.dataset == "nist_spectra":
        train_dataset = Multi_process_NISTDataset(config["train_db_path"],config["dict_path"],config)
        val_dataset = Multi_process_NISTDataset(config["val_db_path"],config["dict_path"],config)
        test_dataset = Multi_process_NISTDataset(config["test_db_path"],config["dict_path"],config)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True,pin_memory=True,collate_fn=Multi_process_batch_collate_fn)
        val_data_loader = torch.utils.data.DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=True,pin_memory=True,collate_fn=Multi_process_batch_collate_fn)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=True,pin_memory=True,collate_fn=Multi_process_batch_collate_fn)

    return train_data_loader, val_data_loader, test_data_loader


def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


'''******* 训练相关  *******'''
def get_optimizer(model, config):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': config["weight_decay"]},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = torch.optim.AdamW(optim_groups, lr=config["lr"], betas=(config["beta1"],config["beta2"]))
    return optimizer


def get_lr(config, it):
    # 1) linear warmup for warmup_iters steps
    if it < config["warmup_iters"]:
        return config["lr"] * it / config["warmup_iters"]
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config["min_lr"] + coeff * (config["lr"] - config["min_lr"])


def get_model(config,args):
    dtype=torch.float32
    if args.model == "SSMoE_MVMS_Spectra_Prediction":
        mol_config = MolConfig(mol_pretrain_pth=config["mol_pretrain_pth"],mol_dict_pth=config["mol_dict_pth"])
        model = SSMoE_MVMS(config,mol_config)
    elif args.model == "SSMoE_MVMS_Retrieval":
        mol_config = MolConfig(mol_pretrain_pth=config["mol_pretrain_pth"],mol_dict_pth=config["mol_dict_pth"])
        model = SSMoE_MVMS_Retrieval(config,mol_config)

    if len(args.gpu.split(',')) > 1:
        model = torch.nn.DataParallel(model,device_ids=[int(x) for x in args.gpu.split(',')])
    model = model.to(dtype)
    model = model.to(device=args.device)
    return model 


# prevent shut down the server, loading the ckpt to continue training
def load_pretrained_model(model,ckpt):
    print(f"Loading pretrained model from {ckpt}, continuing training...")
    checkpoint = torch.load(ckpt, map_location="cpu",weights_only=False)
    missing_keys,unexpeted_keys = model.load_state_dict(checkpoint["model"],strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpeted keys: {unexpeted_keys}")
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    print('Loading the ckpt successfully!')
    return model,iter_num,best_val_loss


def is_seed_finished_in_log(log_file, seed):
    end_msg = f"random seed {seed} Training End!"
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if end_msg in line:
                    return True
    except FileNotFoundError:
        return False
    return False

'''******* 指标相关  *******'''
def l2loss(out,target):
    '''Mean Square Error(MSE)'''
    diff = out-target
    return torch.mean(diff ** 2)

def l1loss(out,target):
    '''Mean Absolute Error(MAE)'''
    return torch.mean(torch.abs(out-target))

def stftloss(out,target):
    preds_sp = torch.stft(out[:,0], n_fft=32, return_complex=False)
    targets_sp = torch.stft(target[:,0], n_fft=32, return_complex=False)
    mse = nn.MSELoss()
    sp_loss = mse(preds_sp, targets_sp)
    return sp_loss


def pearson_correlation(out, target):
    '''
    Pearson's correlation coefficient, used to measure the linear correlation between two variables
    '''
    # 计算均值
    mean_out = torch.mean(out)
    mean_target = torch.mean(target)
    # 计算偏差
    std_out = torch.std(out)
    std_target = torch.std(target)
    # 计算协方差
    covariance = torch.sum((out - mean_out) * (target - mean_target))
    # 计算皮尔逊相关系数
    pearson_corr = covariance / (std_out * std_target * (len(out) - 1))
    return pearson_corr


def criterion(preds, targets):
    mse = nn.MSELoss()
    mse_loss = mse(preds, targets)
    preds_sp = torch.stft(preds[:,0], n_fft=32, return_complex=False)
    targets_sp = torch.stft(targets[:,0], n_fft=32, return_complex=False)
    sp_loss = mse(preds_sp, targets_sp)
    # pairwise_cos_sim(preds, targets)
    # the cos_sim is added after 8k iterations warm up
    # return mse_loss - 0.7*pairwise_cos_sim(preds, targets).log()
    return mse_loss+sp_loss


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True
    

def get_multilabel_performance(all_labels,all_preds):
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)  # 仅完全匹配才算对（更严格）
    return acc,precision,recall,f1