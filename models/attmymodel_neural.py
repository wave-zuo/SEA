import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from models.attlayers import *

class MNN(nn.Module):
    def __init__(self, node_size, nhid1, droput, alpha):
        super(MNN, self).__init__()
        self.attentions = MultiHeadGraphAttention(1, node_size * 2, nhid1, droput)

        self.a1 = nn.Parameter(torch.Tensor(nhid1, 1))
        self.a2 = nn.Parameter(torch.Tensor(nhid1, 1))
        nn.init.xavier_uniform_(self.a1.data)
        nn.init.xavier_uniform_(self.a2.data)
        self.droput = droput
        self.alpha = alpha

    def forward(self, adj_batch, adj_mat, A2, Matrix, b=1, zeros=None):
        fusion_adj = torch.cat([Matrix, A2], dim=1)  # m, n+m

        t0 = self.attentions(fusion_adj, adj_mat)
        t0 = t0.mean(0)

        embedding = t0

        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)

        L_1st = torch.sum(adj_mat * (embedding_norm -
                                     2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                                     + torch.transpose(embedding_norm, dim0=0, dim1=1)))


        a1_out = torch.mm(t0, self.a1)
        a2_out = torch.mm(t0, self.a2)
        t0 = a1_out.expand(-1, adj_mat.shape[0]) + a2_out.expand(-1, adj_mat.shape[0]).permute(1, 0)
        t0 = torch.sigmoid(t0)
        # t0 = torch.sigmoid(torch.mm(t0, t0.t()))

        zeros = torch.zeros_like(adj_batch)
        t0 = torch.where(adj_batch > 0, t0, zeros)
        Matrix = torch.where(adj_batch > 0, Matrix, zeros)
        L_2nd = torch.sum((Matrix - t0) * (Matrix - t0))

        return self.alpha * L_1st, L_2nd, self.alpha * L_1st + L_2nd

    def savector(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0
    def get_res_matrix(self, adj, A2, Matrix):
        fusion_adj = torch.cat([Matrix, A2], dim=1)  # m, n+m
        t0 = self.attentions(fusion_adj, adj)
        t0 = t0.mean(0)
        embed = t0
        a1_out = torch.mm(t0, self.a1)
        a2_out = torch.mm(t0, self.a2)
        res = a1_out.expand(-1, adj.shape[0]) + a2_out.expand(-1, adj.shape[0]).permute(1, 0)
        res = torch.sigmoid(res)

        return res


