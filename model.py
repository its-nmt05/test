from utils import load_nac
from film import FiLM

import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.nac_encoder, self.nac_decoder, self.qt = load_nac()
        self.linear = nn.Linear(dim, dim)
        self.FiLM = FiLM(dim)
        
    def forward(self, wav):
        emb = self.nac_encoder(wav)
        time_avg = emb.mean(dim=-1)
        codes = self.qt.encode(emb)
        emb_r = self.qt.decode(codes)
        time_avg = self.linear(time_avg)
        modulated = self.FiLM(emb_r, time_avg)
        
        out = self.nac_decoder(modulated)
        return out
    

# test
import torchaudio
wv, sr = torchaudio.load('audio.wav')    
wv = wv.unsqueeze(0)
model = Model(dim=128)
torchaudio.save("output.wav", wv.squeeze(0), sr)        
        