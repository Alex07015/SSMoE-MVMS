# early_stopping.py
import numpy as np
import torch
import os
from datetime import datetime


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, trace_func=print, best_val_loss=None, best_iter_num=None, final_iter=800):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_val_loss = best_val_loss
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_iter_num = best_iter_num
        self.trace_func = trace_func
        self.final_iter = final_iter

    def __call__(self, val_loss, model, iter_num, config , optimizer,random_seed, log):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.best_iter_num = iter_num
            self.save_checkpoint(val_loss, model, self.best_iter_num, optimizer, config,random_seed)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.best_iter_num = iter_num
            self.save_checkpoint(val_loss, model, self.best_iter_num, optimizer, config,random_seed)
            self.counter = 0  # Reset counter since improvement occurred
        elif iter_num > self.final_iter:
            self.early_stop = True
            msg = f'Max Iter! Early stopping at iteration {iter_num}. Best iteration: {self.best_iter_num}'
            log.info(msg)
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                msg = f'Out of Patience! Early stopping at iteration {iter_num}. Best iteration: {self.best_iter_num}'
                log.info(msg)
        return self.early_stop

    def save_checkpoint(self, best_val_loss, model, best_iter_num, model_optimizer, config, random_seed=None):
        '''Saves model when validation loss decreases.'''
        if best_iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': model_optimizer.state_dict(),
                'iter_num': best_iter_num,
                'config': config,
                'best_val_loss': best_val_loss
            }
        save_path = os.path.join(os.path.join(config['output_dir'], 'checkpoints', config['type'], 'random_seed_{}.pt'.format(random_seed)))
        torch.save(checkpoint, save_path)
        self.val_loss_min = best_val_loss