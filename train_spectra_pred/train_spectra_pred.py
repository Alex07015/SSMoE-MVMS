import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# 获取当前代码文件绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到sys.path中
sys.path.append(os.path.join(current_dir, ".."))
import torch
from datetime import datetime
from train_utils import *
from args import *
from contextlib import nullcontext
import warnings
warnings.filterwarnings("ignore")
import wandb
from tqdm import tqdm
import shutil
from early_stopping import EarlyStopping


def get_data(loader):
    batch = next(iter(loader))
    return batch

def valdiate(model, val_loader, args, config):
    model.eval()
    step = 0
    epoch_loss = 0.0
    epoch_mae = 0.0
    epoch_stft = 0.0
    mask_ratio = config['mask_ratio']
    for _ in tqdm(range(args.eval_steps),desc='Validation',mininterval=1):
        step += 1
        batch_list = get_data(val_loader)
        src_tokens,src_coord,src_distance,src_edge_type,spectra,smi = batch_list['unimol']["src_tokens"].to(args.device), batch_list['unimol']["src_coord"].to(args.device), batch_list['unimol']["src_distance"].to(args.device), batch_list['unimol']["src_edge_type"].to(args.device), batch_list['unimol']["ir"].to(args.device),batch_list['unimol']["smi"]
        x,edge_index,edge_attr,batch = batch_list['pyg']["x"].to(args.device), batch_list['pyg']["edge_index"].to(args.device), batch_list['pyg']["edge_attr"].to(args.device),batch_list['pyg'].batch.to(args.device)
        loss,_,stftloss,rec_loss = model(src_tokens, src_coord, src_distance, src_edge_type, smi, x, edge_index, edge_attr, batch, spectra, args.device, mask_ratio,preserve_flag=[0,1,2,3])
        if args.dp:
            loss = loss.mean()
            stftloss = stftloss.mean()
            rec_loss = rec_loss.mean()
        epoch_loss += loss.item()
        epoch_stft += stftloss.item()
        epoch_mae += rec_loss.item()

    epoch_loss /= step
    epoch_stft /= step
    epoch_mae /= step
    model.train()
    return epoch_loss,epoch_stft,epoch_mae


def train_and_valid(model, iter_num, train_loader, val_loader, model_optimizer, base_logger,  config, early_stopping, seed, args):
    device_type = config['device_type']
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    while True:
        iter_num += 1
        model.train()
        lr = get_lr(config, iter_num) if config['decay_lr'] else config['lr']
        for param_group in model_optimizer.param_groups:
            param_group['lr'] = lr        
        if args.wandb:
            wandb.init(
                project="Qformer",  # 项目名称
                name=args.wandb_name,     # 实验名称
                config=config,
                # resume="allow",
                # id="59lghh8m",
        )
        if iter_num % args.eval_interval == 0:
            with torch.no_grad():
                val_loss,val_spectra_loss,val_rec = valdiate(model,val_loader,args,config)
                msg = f"iter {iter_num} val loss: {val_loss} val_spectra_loss: {val_spectra_loss} val_rec: {val_rec} lr: {lr}"
                log_and_print(base_logger, msg)
                # early stopping
                early_stop = early_stopping(val_loss, model, iter_num, config, model_optimizer, random_seed=seed,log=base_logger)
                if early_stop:
                    msg = f"random seed {seed} Training End!"
                    log_and_print(base_logger, msg)
                    break
            if args.wandb:
                wandb.log({"val_loss": val_loss,"lr":lr})
        if iter_num == 0:
            break

        accumulate_loss = 0.0
        accumulate_stft = 0.0
        accumulate_mae = 0.0
        mask_ratio = config['mask_ratio']
        # 训练num_steps次数
        for _ in tqdm(range(args.num_steps),desc="Training step",mininterval=1):
            batch_list = get_data(train_loader)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            with ctx:
                src_tokens,src_coord,src_distance,src_edge_type,spectra,smi = batch_list['unimol']["src_tokens"].to(args.device), batch_list['unimol']["src_coord"].to(args.device), batch_list['unimol']["src_distance"].to(args.device), batch_list['unimol']["src_edge_type"].to(args.device), batch_list['unimol']["ir"].to(args.device),batch_list['unimol']["smi"]
                x,edge_index,edge_attr,batch = batch_list['pyg']["x"].to(args.device), batch_list['pyg']["edge_index"].to(args.device), batch_list['pyg']["edge_attr"].to(args.device),batch_list['pyg'].batch.to(args.device)
                loss,_,stftloss,recloss = model(src_tokens, src_coord, src_distance, src_edge_type, smi, x, edge_index, edge_attr, batch, spectra, args.device, mask_ratio)
                if args.dp:
                    loss = loss.mean()
                    stftloss = stftloss.mean()
                    recloss = recloss.mean()
                scaler.scale(loss).backward()
            accumulate_loss += loss.item()
            accumulate_stft += stftloss.item()
            accumulate_mae += recloss.item()
            if config['grad_clip'] != 0.0:
                scaler.unscale_(model_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(model_optimizer)
            scaler.update()
            model_optimizer.zero_grad(set_to_none=True)   
        accumulate_loss /= args.num_steps
        accumulate_stft /= args.num_steps
        accumulate_mae /= args.num_steps
        if args.wandb:
            wandb.log({"train_loss":accumulate_loss})
        msg = f"iter: {iter_num}, Average train_loss: {accumulate_loss} Average stft_loss: {accumulate_stft} Average rec_loss: {accumulate_mae}"
        log_and_print(base_logger, msg)
    wandb.finish()
    shutil.rmtree("wandb", ignore_errors=True)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    set_folders(config)
    log = logger(config)
    checkpoint_dir =  os.path.join(os.path.join(config['output_dir'], 'checkpoints', config['type']))
    log_file = os.path.join('exp',config['output_dir'], '{}.log'.format(config['type']))
    for i in range(config["num_runs"]):
        seed = args.seed + i
        ckpt_path = os.path.join(checkpoint_dir, f"random_seed_{seed}.pt")
        if is_seed_finished_in_log(log_file, seed):
            print(f"Seed {seed} 已在日志中结束训练，跳过")
            continue
        init_seed(seed)
        msg = f"Running with seed {seed}, Start Training!"
        log_and_print(log, msg)
        print(f"config pth:{args.conf}")
        msg = f"batch size: {config['batch_size']}, num_steps: {args.num_steps}, eval_interval: {args.eval_interval}, eval_steps: {args.eval_steps}, cuda/cpu:{args.device} gpus: {args.gpu}, model: {args.model}"
        log_and_print(log, msg)
        current_path = os.path.abspath(__file__)
        msg = f"当前脚本路径：{current_path}" 
        log_and_print(log, msg)
        train_loader, val_loader,test_loader = get_dataloaders(config,args)
        model = get_model(config,args)
        if args.pretrained and os.path.exists(ckpt_path):
            ckpt = r""
            model,iter_num,best_val_loss = load_pretrained_model(model, ckpt)
        else:
            iter_num = 0
            best_val_loss = float("inf")
        optimizer = get_optimizer(model, config)
        early_stopping = EarlyStopping(patience=20, delta=0, trace_func=print, best_val_loss=best_val_loss, best_iter_num=iter_num,final_iter=1500)
        train_and_valid(model, iter_num, train_loader, val_loader, optimizer, log, config, early_stopping, seed, args)