import argparse
import yaml
import os
from train_utils import get_logger

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='/home/zjh/project/Mol_GPT/config/config.yaml', help='Configuration yaml file')  # keep second
    parser.add_argument('--seed', type=int, default=42, help='random seed for training')
    parser.add_argument('--num_steps', default=50, type=int, help='Maximum number of gradient steps.')

    parser.add_argument('--save',type=bool,required=False,default=True, help='if set true, save the best model')
    parser.add_argument('--pretrained',type=bool,required=False,default=False,help='whether to use pretrained model for training')
    parser.add_argument('--eval_interval',required=False,default=5,type=int,help='the interval to logger the val model')
    parser.add_argument('--eval_steps',required=False,default=50,type=int,help='the steps to evaluate the model')
    parser.add_argument('--device',type=str,required=False,default='cuda:0',help="Use CPU or GPU for training")
    parser.add_argument('--dtype',type=str,default='float32',help='data type for training')
    parser.add_argument('--gpu',type=str,required=False,default='0',help='id of gpu device(s) to be used')
    parser.add_argument('--dp',type=str,default=False,help='dp for training')
    parser.add_argument('--wandb',type=str,default=False,help='Whether to use wandb for logging')
    parser.add_argument('--wandb_name',type=str,default='Test',help='the name of config file to be saved')
    parser.add_argument('--model',type=str,default='mol-qformer',help='training mode name')
    parser.add_argument('--dataset', type=str, default='qm9s', help='dataset: qm9s or uspto')  # keep second
    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args


def load_config(args):
    # load config file into dict() format
    with open(args.conf, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_folders(config):
    # create the output folder (name of experiment) for storing model result such as logger information
    if not os.path.exists(os.path.join('exp',config['output_dir'])):
        os.mkdir(os.path.join('exp',config['output_dir']))
    # create the root folder for storing checkpoints during training
    if not os.path.exists(os.path.join('exp',config['output_dir'], 'checkpoints')):
        os.mkdir(os.path.join('exp',config['output_dir'], 'checkpoints'))
    # create the subfolder for storing checkpoints based on the model type
    if not os.path.exists(os.path.join('exp',config['output_dir'], 'checkpoints', config['type'])):
        os.mkdir(os.path.join('exp',config['output_dir'], 'checkpoints', config['type']))
    with open(os.path.join('exp',config['output_dir'], config['save_config_name']), 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)    

def logger(config):
    file_name = os.path.join('exp',config['output_dir'], '{}.log'.format(config['type']))
    base_logger = get_logger(file_name, config['log_title'])
    return base_logger