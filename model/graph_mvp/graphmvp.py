from .molecular_gnn_model import graphmvp
import torch
import torch.nn as nn
import torch.nn.functional as F


def criterion(preds, targets):
    mse = nn.MSELoss()
    preds_sp = torch.stft(preds, n_fft=32, return_complex=False)
    targets_sp = torch.stft(targets, n_fft=32, return_complex=False)
    sp_loss = mse(preds_sp, targets_sp)
    return sp_loss


class Graphmvp_pred(nn.Module):
    def __init__(self,config):
        super(Graphmvp_pred, self).__init__()
        self.model = graphmvp(num_layer=config['graphmvp_num_layer'], emb_dim=config['graphmvp_emb_dim'], num_tasks=config['num_tasks'], JK=config['graphmvp_JK'], graph_pooling=config['graphmvp_graph_pooling'])
        self.model.from_pretrained(config["pretrained_pth"])
        self.task = config["task"]
        self.task_type = config["task_type"]
        if self.task == "spectra":
            self.out = nn.Linear(300, 512)
        elif self.task == 'bbbp':
            self.out = nn.Linear(300, 1)
        elif self.task == 'bace':
            self.out = nn.Linear(300, 1)
        elif self.task == 'tox21':
            self.out = nn.Linear(300, 12)
        elif self.task == 'sider':
            self.out = nn.Linear(300, 27)
        elif self.task == 'clintox':
            self.out = nn.Linear(300, 2)
        elif self.task == 'esol':
            self.out = nn.Linear(300, 1)
        elif self.task == 'freesolv':
            self.out = nn.Linear(300, 1)
        elif self.task == 'lipo':
            self.out = nn.Linear(300, 1)
        elif self.task == 'muv':
            self.out = nn.Linear(300, 17)
        elif self.task == 'hiv':
            self.out = nn.Linear(300, 1)
        elif self.task == 'toxcast':
            self.out = nn.Linear(300, 617)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,x, edge_index, edge_attr, batch, spectra=None):
        repr, _, _, _= self.model(x, edge_index, edge_attr, batch)
        pred = self.out(repr)
        if self.task == "spectra" and spectra is not None:
            stftloss = criterion(pred, spectra.reshape(pred.shape))
            rec_loss = nn.L1Loss()(pred, spectra.reshape(pred.shape))
            loss = stftloss * 0.1 + rec_loss
            return loss, pred, stftloss, rec_loss
        else:
            return pred
            


