import torch.nn as nn

class FiLM(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.f_c = nn.Linear(dim, dim)
        self.h_c = nn.Linear(dim, dim)
        
    def forward(self, x, cond):
        gamma = self.f_c(cond).unsqueeze(-1)
        beta = self.h_c(cond).unsqueeze(-1)
        return gamma * x + beta
    
        