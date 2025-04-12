import torch.nn as nn

class FiLM(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.f_c = nn.Linear(dim, dim)
        self.h_c = nn.Linear(dim, dim)
        
    def forward(self, x):
        gamma = self.f_c(x)
        beta = self.h_c(x)
        return gamma * x + beta
    
        