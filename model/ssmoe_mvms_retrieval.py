from pyexpat import model
from sympy import im
import sys
import torch.nn as nn
import torch
from .SMILES.smilesEncoder import SMiles_Encoder
import math
import torch.nn.functional as F
from .graph_mvp.molecular_gnn_model import graphmvp
from .unimol.unimol import UniMolModel
from .moe import DeepseekMoE
from torch.nn.functional import leaky_relu
from copy import deepcopy
from .spectra_encoder import SpectraTransformer

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))  # [D]
    
    def forward(self, token_embeddings, attention_mask):
        """
        token_embeddings: [B, L, D]
        attention_mask: [B, L]
        """
        B, L, D = token_embeddings.size()

        # [B, L]
        attn_scores = (token_embeddings @ self.query) / (D ** 0.5)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, L]

        # Apply attention
        pooled = torch.bmm(attn_weights.unsqueeze(1), token_embeddings)  # [B, 1, D]
        return pooled.squeeze(1)  # [B, D]
    

class Embedding(nn.Module):
    def __init__(self,block_size, embed_dim, bias, dropout):
        super().__init__()
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.LayerNorm = LayerNorm(embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "position_ids",
            torch.arange(block_size).expand((1, -1)),
        )
        self.modality_embeddings = nn.Embedding(2, embed_dim)

    def forward(self, smiles_feats=None, graph_feats=None, position_ids=None):
        b = smiles_feats.size(0) if smiles_feats is not None else graph_feats.size(0) # type: ignore
        # assert b == graph_feats.size(0), "batch size of mol_3d_feats and smiles_feats should be the same" # type: ignore
        device = smiles_feats.device if smiles_feats is not None else graph_feats.device # type: ignore
        if smiles_feats is not None:
            smiles_feats_seq_length = smiles_feats.size(1)
        else:
            smiles_feats_seq_length = 0
        if graph_feats is not None:
            graph_feats_seq_length = graph_feats.size(1) # type: ignore
        else:
            graph_feats_seq_length = 0
        total_length = graph_feats_seq_length + smiles_feats_seq_length # type: ignore
        if position_ids is None:
            position_ids = self.position_ids[:, :total_length].clone() # type: ignore
        
        pos_embeddings = self.position_embedding(position_ids)  # [1, seq_length, D]
        if smiles_feats is not None and graph_feats is not None:
            token = torch.cat([graph_feats, smiles_feats], dim=1)  # type: ignore # [B, seq_length, D]
        elif graph_feats is None:
            token = smiles_feats  # type: ignore # [B, seq_length, D]
        elif smiles_feats is None:
            token = graph_feats  # type: ignore # [B, seq_length, D]
        modality_ids = torch.cat([
            torch.zeros((b, graph_feats_seq_length), dtype=torch.long),    # graph
            torch.ones((b, smiles_feats_seq_length), dtype=torch.long),     # smiles
        ], dim=1).to(device)
        modality_embed = self.modality_embeddings(modality_ids)
        embeddings = token + pos_embeddings + modality_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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

    def __init__(self, embed_dim, n_head, bias, dropout,mlp_sparse,num_experts,num_experts_per_tok,n_shared_experts,aux_loss_alpha,seq_aux,norm_topk_prob,dense_idx, idx):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias=bias)
        self.attn = SelfAttention(embed_dim, n_head, bias, dropout)
        self.ln_2 = LayerNorm(embed_dim, bias=bias)
        self.mlp_sparse = mlp_sparse
        if idx > dense_idx:
            if self.mlp_sparse:
                self.mlp = DeepseekMoE(
                    moe_intermediate_size=embed_dim*2,
                    hidden_size=embed_dim,
                    n_shared_experts=n_shared_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    n_routed_experts=num_experts,
                    norm_topk_prob=norm_topk_prob,
                    aux_loss_alpha=aux_loss_alpha,
                    seq_aux=seq_aux
                )
            else:
                self.mlp = MLP(embed_dim, bias, dropout)
        else:
            self.mlp = MLP(embed_dim, bias, dropout)


    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        # Self-attention
        y, attn_weights = self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + y
        # Feed-forward
        residual = x
        if isinstance(self.mlp, DeepseekMoE):
            x,top_k = self.mlp(self.ln_2(x))
        else:
            x = self.mlp(self.ln_2(x))
            top_k = None
        x = residual + x
        return x, attn_weights,top_k

class conv_block(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        return x



class SSMoE_MVMS_Retrieval(nn.Module):
    def __init__(self, config,mol_config):
        super().__init__()
        # ========================SMILES ENCODER========================
        self.molformer = SMiles_Encoder(config['molformer_pth'])
        self.chemberta = SMiles_Encoder(config['chemberta_pth'])
        self.graphmvp = graphmvp(
            num_layer=config["graphmvp_num_layer"], 
            emb_dim=config["graphmvp_emb_dim"],
            num_tasks=config["graphmvp_num_tasks"],
            JK=config["graphmvp_JK"],
            graph_pooling=config["graphmvp_graph_pooling"],
            molecule_node_model=None  # You can pass a specific GNN model if needed
        )
        self.graphmvp.from_pretrained(config["graphmvp_pretrain_pth"])
        self.unimol = UniMolModel(mol_config)
        self.unimol.load_pretrained_weights(mol_config.mol_pretrain_pth)
        if config["unfreeze_model"]:
            print(f"Freezing the all models!")
            for param in self.molformer.parameters():
                param.requires_grad = False
            for param in self.chemberta.parameters():
                param.requires_grad = False
            for param in self.graphmvp.parameters():
                param.requires_grad = False
            for param in self.unimol.parameters():
                param.requires_grad = False
        self.graph_mvp_project = nn.Linear(config['graphmvp_emb_dim'], config['embed_dim'])
        self.unimol_project = nn.Linear(config['unimol_emb_dim'], config['embed_dim'])
        self.molformer_project = nn.Linear(config['molformer_embed_dim'], config['embed_dim'])
        self.chemberta_project = nn.Linear(config['chemberta_embed_dim'], config['embed_dim'])
        self.transformer = nn.ModuleList([Block(config["embed_dim"],config["n_head"],config["bias"],config["dropout"],config["mlp_sparse"],config["num_experts"], 
                                                config["num_experts_per_tok"],config["n_shared_experts"],config["aux_loss_alpha"],config["seq_aux"],config["norm_topk_prob"],config["dense_idx"], j) for j in range(config["n_layer"])])
        self.embedding = Embedding(config["block_size"], config["embed_dim"], config["bias"], config["dropout"])
        self.attention_pooling = AttentionPooling(config["embed_dim"])
        self.total_model_nums = config["total_model_nums"]
        self.mol_project = nn.Linear(config["embed_dim"],config["cl_hidden_size"])
        self.spectra_project = nn.Linear(config["spectra_embed_dim"],config["cl_hidden_size"])
        self.temp = nn.Parameter(torch.ones([]) * config["temperature"])
        self.spectra_encoder = SpectraTransformer(config)
        print(f"total model nums: {self.total_model_nums}")
        print("number of Spectra_Mol_cl_DeepseekMOE parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=encoder_attention_mask.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        return encoder_extended_attention_mask

    def _build_model_token_mask(self, model_lengths,device):
        total_len = sum(model_lengths)
        masks = torch.zeros((self.total_model_nums, total_len), dtype=torch.float32,device=device)
        start = 0
        for i, l in enumerate(model_lengths):
            masks[i, start:start + l] = 1
            start += l
        return masks  # shape: [6, total_token_length]

    def apply_group_mask(self, attention_mask, group, model_token_mask, num_models,device):
        if group.numel() == 0:
            assert group == 0, "Group should not be empty!"
            return
        rand_indices = torch.stack([torch.randperm(self.total_model_nums)[:num_models] for _ in range(len(group))]).to(device)
        selected_model_masks = model_token_mask[rand_indices]  # [B, num_models, total_len]
        combined_mask = 1.0 - torch.clamp(selected_model_masks.sum(dim=1), max=1.0)
        attention_mask[group] = attention_mask[group] * combined_mask
        return attention_mask

    def generate(self, attention_mask, model_lengths, mask_ratio=0.5):
        b = attention_mask.size(0)
        device = attention_mask.device
        # Step 2: 要掩码的样本索引
        num_mask_samples = int(b * mask_ratio)
        mask_sample_indices = torch.randperm(b, device=device)[:num_mask_samples]
        model_token_mask = self._build_model_token_mask(model_lengths,device)
        num_2 = int(num_mask_samples * 0.5)

        group_2 = mask_sample_indices[:num_2]
        group_3 = mask_sample_indices[num_2:]

        attention_mask = self.apply_group_mask(attention_mask, group_2, model_token_mask,
                                               num_models=2, device=device)
        attention_mask = self.apply_group_mask(attention_mask, group_3, model_token_mask
                                                , num_models=3, device=device)
        return attention_mask
    
    def encode_except(self, preserve_flag):
        if isinstance(preserve_flag, (list, tuple, set)):
            return [i for i in range(self.total_model_nums) if i not in preserve_flag]
        else:  # assume it's a single int
            return [i for i in range(self.total_model_nums) if i != preserve_flag]
    
    def inference_mask(self,attention_mask,model_lengths,preserve_flag=None):
        device = attention_mask.device
        mask_model_index = self.encode_except(preserve_flag) if preserve_flag is not None else list(range(self.total_model_nums))
        model_token_mask = self._build_model_token_mask(model_lengths,device)
        selected_model_masks = model_token_mask[mask_model_index]  # [B, num_models, total_len]
        combined_mask = 1.0 - torch.clamp(selected_model_masks.sum(dim=0), max=1.0)
        combined_mask = combined_mask.repeat(attention_mask.size(0), 1).to(device)  # [B, num_models]
        attention_mask = attention_mask * combined_mask
        return attention_mask

    
    def forward(self, src_tokens, src_coord, src_distance, src_edge_type, smi, x, edge_index, edge_attr,batch, spectra, device, mask_ratio,preserve_flag=None):
        b = spectra.size(0)
        # ========================SMILES ENCODER========================
        molformer_last_hidden_state, _, molformer_attention_mask = self.molformer.forward_feats(smi,device)
        chemberta_last_hidden_state, _, chemberta_attention_mask = self.chemberta.forward_feats(smi,device)
        molformer_last_hidden_state = self.molformer_project(molformer_last_hidden_state)
        chemberta_last_hidden_state = self.chemberta_project(chemberta_last_hidden_state)
        # print(f"smiles token length: molformer: {molformer_last_hidden_state.size()}, chemberta: {chemberta_last_hidden_state.size()}, bertsmiles: {bertsmiles_last_hidden_state.size()}")
        smiles_feats = torch.cat([molformer_last_hidden_state,chemberta_last_hidden_state], dim=1)
        # ========================Graph ENCODER========================
        unimol_dict = self.unimol(src_tokens,src_distance,src_coord,src_edge_type) # type: ignore
        unimol_atom_feats = unimol_dict["all_repr"]
        unimol_attention_mask = unimol_dict["padding_mask"]
        _, graph_mvp_feats, graphmvp_attention_mask, _ = self.graphmvp(x, edge_index, edge_attr, batch) # type: ignore
        unimol_feats = self.unimol_project(unimol_atom_feats)
        graphmvp_feats = self.graph_mvp_project(graph_mvp_feats)
        graph_feats = torch.cat([unimol_feats, graphmvp_feats], dim=1)
        attention_mask = torch.cat([unimol_attention_mask,graphmvp_attention_mask,molformer_attention_mask,chemberta_attention_mask], dim=1)
        model_length = [unimol_attention_mask.size(1), graphmvp_attention_mask.size(1),molformer_attention_mask.size(1),chemberta_attention_mask.size(1)]
        if preserve_flag is None and mask_ratio != 0.0:
            attention_mask = self.generate(attention_mask,model_length,mask_ratio)
        else:
            # print(f"During the inference step!")
            attention_mask = self.inference_mask(attention_mask,model_length,preserve_flag)
            # print(attention_mask[0,:]) # type: ignore
        attention_mask = attention_mask.int() # type: ignore
        encoder_extended_attention_mask = self.invert_attention_mask(attention_mask)
        x = self.embedding(smiles_feats=smiles_feats,graph_feats=graph_feats)
        all_attn_weights = []
        all_top_k = []
        for block in self.transformer:
            x,attn,top_k = block(x,attention_mask=encoder_extended_attention_mask)
            all_attn_weights.append(attn)
            all_top_k.append(top_k)
        pooler_feature = self.attention_pooling(x, attention_mask)
        mol_feats = F.normalize(self.mol_project(pooler_feature), dim=-1)
        spectra_feature = self.spectra_encoder(spectra)
        spectra_feats = F.normalize(self.spectra_project(spectra_feature), dim=-1)
        targets = torch.arange(b, device=device)
        temp = self.temp.exp()
        logits = temp *  mol_feats @ spectra_feats.t() 
        clip_loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets.t())) / 2
        return clip_loss,mol_feats,spectra_feats

