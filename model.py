from utils import load_nac
from film import FiLM

import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.nac_encoder, self.nac_decoder, self.qt = load_nac()
        self.linear == nn.Linear(dim)
        self.FiLM = FiLM(dim)
        
    def forward(self, wav):
        emb = self.nac_encoder(wav)
        time_avg = emb.mean(dim=-1)
        codes = self.qt.encoder(emb)
        emb_r = self.qt.decoder(codes)
        
        time_avg = self.linear(time_avg)
        modulated = self.FiLM(emb_r, time_avg)
        
        out = self.nac_decoder(modulated)
        return out
        
        