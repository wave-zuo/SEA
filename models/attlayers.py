import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout=0.1, bias=True, act=F.leaky_relu):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        # self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        # self.dropout = attn_dropout
        self.act = act

        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w.data)
        nn.init.xavier_uniform_(self.a_src.data)
        # nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        n = h.size(0) # h is of size n x f_in
        h_prime = torch.matmul(h.unsqueeze(0), self.w) #  n_head x n x f_out
        attn_src = torch.bmm(h_prime, self.a_src) # n_head x n x 1
        attn_dst = torch.bmm(h_prime, self.a_src) # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1) # n_head x n x n

        attn = self.leaky_relu(attn)
        # attn = torch.sigmoid(attn)

        attn.data.masked_fill_((1 - adj).bool(), float(-9e15))
        attn = self.softmax(attn) # n_head x n x n
        attn = self.dropout(attn)
        # attn = F.dropout(attn, self.dropout, training=self.training)
        output = torch.bmm(attn, h_prime) # n_head x n x f_out

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)
    def __repr__(self):
        return self.__class__.__name__