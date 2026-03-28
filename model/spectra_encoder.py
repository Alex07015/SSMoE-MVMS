import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head, bias, dropout):
        super().__init__()
        assert embed_dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = embed_dim
        self.dropout = dropout


    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attention_mask is not None:
            att = att + attention_mask
        att_weights = F.softmax(att, dim=-1)
        att = self.attn_dropout(att_weights)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y,att_weights


class MLP(nn.Module):

    def __init__(self, embed_dim, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):

    def __init__(self, embed_dim, n_head, bias, dropout):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias=bias)
        self.attn = SelfAttention(embed_dim, n_head, bias, dropout)
        self.ln_2 = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(embed_dim, bias, dropout)

    def forward(self, x, attention_mask=None):
        # Self-attention
        y, attn_weights = self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + y
        # Feed-forward
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights



class SpectraTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["spectra_block_size"] is not None, "spectra_block_size must be defined"
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(config["spectra_n_patchsize"], config["spectra_embed_dim"],bias=False),
            wpe = nn.Embedding(config["spectra_block_size"], config["spectra_embed_dim"]),
            drop = nn.Dropout(config["spectra_dropout"]),
            h = nn.ModuleList([Block(config["spectra_embed_dim"], config["spectra_n_head"], config["spectra_bias"], config["spectra_dropout"]) for _ in range(config["spectra_n_layer"])]),
            ln_f = LayerNorm(config["spectra_embed_dim"], bias=config["spectra_bias"]),
        ))
        self.lm_head = nn.Linear(config["spectra_embed_dim"], config["spectra_n_patchsize"], bias=False)
        self.linear_unpatch = nn.Linear(config["spectra_embed_dim"],config["spectra_n_patchsize"])
        self.apply(self._init_weights)
        self.num_of_patch_per_spectra = config["signal_length"] // config["spectra_n_patchsize"]

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config["spectra_n_layer"]))
        # report number of parameters
        print("number of spectra_transformer parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def load_pretrain_pth(self, path, strict=False):
        if path is not None:
            print(f"loading pretrained weights from {path}")
            state_dict = torch.load(path, map_location="cpu")
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            try:
                if "module." in state_dict:
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    missing_keys,unexpeted_keys = self.load_state_dict(state_dict, strict=strict)
                    print("missing_keys: ", missing_keys)
                    print("unexpeted_keys: ", unexpeted_keys)
                else:
                    missing_keys,unexpeted_keys = self.load_state_dict(state_dict, strict=strict)
                    print("missing_keys: ", missing_keys)
                    print("unexpeted_keys: ", unexpeted_keys)
            except RuntimeError as e:
                    raise e
            
    def patchify(self, x):
        # input x of shape (b, n, l) is a batch of n spectra, each of length l
        # we want to convert l-length spectrum into a batch of n patches of size n_patch_size (l = n_patch_size * n_patch)
        # this is done by splitting the spectra into n_patch_size chunks along the length dimension
        # and stacking them up as a new batch dimension
        b, n, l = x.size()
        x = x.view(b, n, self.num_of_patch_per_spectra, self.config["spectra_n_patchsize"])
        x = x.contiguous().view(b, self.num_of_patch_per_spectra * n, self.config["spectra_n_patchsize"])
        return x  

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, spectra):
        device = spectra.device
        b, n, l = spectra.size()
        patches = self.patchify(spectra) # shape (b, t, n_patchsize)
        t = n * l // self.config["spectra_n_patchsize"]
        assert t  <= self.config["spectra_embed_dim"], f"Cannot forward sequence of length {t}, block size is only {self.config['spectra_embed_dim']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # forward the GPT model itself
        inputs = patches.clone()
        tok_emb = self.transformer.wte(inputs) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x,_ = block(x)
        x = self.transformer.ln_f(x)
        x = torch.mean(x, dim=1)
        return x
