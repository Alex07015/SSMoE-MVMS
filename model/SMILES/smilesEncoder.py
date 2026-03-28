from pyexpat import model
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SMiles_Encoder(nn.Module):
    def __init__(self,model_pth):
        super(SMiles_Encoder, self).__init__()
        print("model name: %s" % (model_pth))
        self.model,self.tokenizer = self.build_model(model_pth)
        print("number of SMiles_Encoder parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def build_model(self,path):
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return model,tokenizer
    
    def forward(self, smiles_data,device):
        inputs = self.tokenizer(smiles_data, max_length=512, padding=True, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        return outputs
    
    def forward_feats(self, smiles_data,device):
        inputs = self.tokenizer(smiles_data, max_length=512, padding=True, return_tensors="pt").to(device)
        attention_mask = inputs["attention_mask"]
        outputs = self.model(**inputs)
        return outputs["last_hidden_state"],outputs["pooler_output"],attention_mask
