import torch
import torch.optim as optim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data.dataloader import DataLoader
from data import dataset
from models import attmymodel_neural
import numpy as np
import pandas as pd
import networkx as nx
import time
import math
# epochs=500 no dropout lr=0.01,per 100 *0.9 alpha=1 nu1=1e-6 nu2=1e-5
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--epochs', default=300, type=int,
                        help='The training epochs of SDNE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-5, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=1., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-6, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=5e-5, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=300, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--nhid1', default=128, type=int,
                        help='The embedding dim')
    parser.add_argument('--step_size', default=100, type=int,
                        help='The step size for lr')
    parser.add_argument('--gamma', default=0.9, type=int,
                        help='The gamma for lr')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    allresults = []
    allcorrs = []
    for i in range(10):
        Matrix, Adj, Node = dataset.Read_graph('data/neural/neural2'+str(i), 296)
        samdata = pd.read_csv('data/neural/neuralsam'+str(i)+'.csv')

        # we know all topology info, adj_mat is for GAT to do aggregation
        adj_mat = Adj.clone()
        for e in samdata.iloc[:, :-1].values:
            adj_mat[e[0], e[1]] = 1
            adj_mat[e[1], e[0]] = 1
        adj_mat = adj_mat.cuda()

        Matrix = Matrix.cuda()
        Adj = Adj.cuda()
        A2 = torch.matmul(adj_mat, adj_mat).cuda()
        mask1 = torch.eye(Node).cuda()
        zeros = torch.zeros_like(A2).cuda()
        A2 = torch.where(mask1 > 0, zeros, A2)
        A2 = (A2 - torch.min(A2)) / (torch.max(A2) - torch.min(A2))
        adj_mat = adj_mat + torch.eye(Adj.shape[0], device=device)

        model = attmymodel_neural.MNN(Node, args.nhid1, args.dropout, args.alpha)
        opt = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

        model = model.cuda()
        model.train()
        for epoch in range(1, args.epochs + 1):
            loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
            adj_batch = Adj
            matrix = Matrix
            a2 = A2
            opt.zero_grad()
            L_1st, L_2nd, L_all = model(adj_batch, adj_mat, a2, matrix)
            L_reg = 0
            for param in model.parameters():
                L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
            Loss = L_all + L_reg
            Loss.backward()
            opt.step()
            loss_sum += Loss
            loss_L1 += L_1st
            loss_L2 += L_2nd
            loss_reg += L_reg
            scheduler.step(epoch)
            print("loss for epoch %d is: loss_sum is %f, loss_L1 is %f, loss_L2 is %f, loss_reg is %f" % (
                epoch, loss_sum, loss_L1, loss_L2, loss_reg))

        model.eval()
        model = model.to("cpu")
        Adj = Adj + mask1
        Adj = Adj.cpu()
        adj_mat = adj_mat.cpu()
        A2 = A2.cpu()
        Matrix = Matrix.cpu()

        res_a = model.get_res_matrix(adj_mat, A2, Matrix)
        res = res_a
        res = res.cpu().detach().numpy()

        adj = Adj.numpy()
        res = (res + res.T)/2

        samdata = pd.read_csv('data/neural/neuralsam'+str(i)+'.csv')
        samdata['2'] = np.exp(-1/samdata['2'])
        samdata = samdata.values
        rmse = 0
        pre = []
        for j in range(len(samdata)):
            rmse = rmse + ((samdata[j, 2] - res[int(samdata[j, 0]), int(samdata[j, 1])]) ** 2)
            pre.append(res[int(samdata[j, 0]), int(samdata[j, 1])])
        rmse = math.sqrt(rmse / len(samdata))

        allresults.append(rmse)
        print('RMSE = ' + str(rmse))

        pre = pd.Series(pre)
        truev = pd.Series(samdata[:, -1])
        cor = pre.corr(truev)
        print('corr = ' + str(cor))
        allcorrs.append(cor)
    print('mean='+str(np.mean(allresults))+', std=' + str(np.std(allresults)))
    print(allresults)
    print(allcorrs)
    print('cor mean='+str(np.mean(allcorrs))+', std=' + str(np.std(allcorrs)))
