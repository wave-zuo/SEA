import networkx as nx
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import torch
import scipy.sparse as sp
import pandas as pd

def Read_graph(file_name, nodes):
    data = np.loadtxt(file_name)
    edge = data[:, :-1].astype(np.int32)
    min_node, max_node = edge.min(), edge.max()
    Node = nodes

    Adj = np.zeros([Node, Node], dtype=np.int32)
    Matrix = np.zeros([Node, Node])
    for i in range(edge.shape[0]):
        if min_node == 0:
            Adj[edge[i][0], edge[i][1]] = 1
            Adj[edge[i][1], edge[i][0]] = 1
            Matrix[edge[i][0], edge[i][1]] = np.exp(-1/data[i][2])
            Matrix[edge[i][1], edge[i][0]] = np.exp(-1/data[i][2])
        else:
            Adj[edge[i][0] - 1, edge[i][1] - 1] = 1
            Adj[edge[i][1] - 1, edge[i][0] - 1] = 1
            Matrix[edge[i][0]-1, edge[i][1]-1] = np.exp(-1/data[i][2])
            Matrix[edge[i][1]-1, edge[i][0]-1] = np.exp(-1/data[i][2])
    Adj = torch.FloatTensor(Adj)
    Matrix = torch.FloatTensor(Matrix)
    return Matrix, Adj, int(Node)

def Read_graph_us2(file_name):
    f = open(file_name, 'r')
    Node = f.readline()
    Node = int(Node)
    f.close()

    data = np.loadtxt(file_name, skiprows=1)
    edge = data[:, :-1].astype(np.int32)

    Adj = np.zeros([Node, Node], dtype=np.int32)
    Matrix = np.zeros([Node, Node])
    for i in range(edge.shape[0]):
        Adj[edge[i][0], edge[i][1]] = 1
        Adj[edge[i][1], edge[i][0]] = 1
        Matrix[edge[i][0], edge[i][1]] = np.exp(-1/data[i][2])
        Matrix[edge[i][1], edge[i][0]] = np.exp(-1/data[i][2])
    Adj = torch.FloatTensor(Adj)
    Matrix = torch.FloatTensor(Matrix)
    return Matrix, Adj, Node

class Dataload(data.Dataset):

    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node
    def __getitem__(self, index):
        return index
        # adj_batch = self.Adj[index]
        # adj_mat = adj_batch[index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = self.Beta
        # return adj_batch, adj_mat, b_mat
    def __len__(self):
        return self.Node

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])*np.exp(-1)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
if __name__ == '__main__':
    G, Adj, Node = Read_graph('neural/neural20')
    Data = Dataload(Adj, Node)
    Test = DataLoader(Data, batch_size=20, shuffle=True)
    for index in Test:
        print(index)
        # adj_batch = Adj[index]
        # adj_mat = adj_batch[:, index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = 5
        # print(b_mat)