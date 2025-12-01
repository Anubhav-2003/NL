import torch
from torch import nn as nn
import torch.nn.functional as F

class QKProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self,Q, K):
        scores = torch.einsum("btd,bsd->bts", K, K)
        L = Q.size(1)
        mask = torch.tril(torch.ones(L, L, device=Q.device))
        scores = scores * mask.unsqueeze(0)
        Q_proj = torch.einsum('bts,bsd->btd', scores, Q)

        return Q + Q_proj