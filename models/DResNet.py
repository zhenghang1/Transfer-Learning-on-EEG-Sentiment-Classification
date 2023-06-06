import torch
import torch.nn as nn
import sys
sys.path.append("..")
from models.ADDA import MLP


class ResEncoder(nn.Module):
    def __init__(self, shared_units, domain_units, dropout):
        super().__init__()
        self.dom_shared_encoder = MLP(shared_units, dropout)
        self.dom_bias_encoder = nn.ModuleList([MLP(domain_units, dropout) for _ in range(14)])
    
    def forward(self, X, index=None, is_train=False):
        if is_train:
            return self.dom_shared_encoder(X) + self.dom_bias_encoder[index](X)
        else:
            return self.dom_shared_encoder(X)