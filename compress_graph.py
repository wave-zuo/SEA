import pandas as pd
import numpy as np
import networkx as nx
import random


# create train graph
def read_graph(edges, nodes):
    g = nx.Graph()
    for i in range(nodes):
        g.add_node(i)
    for edge in edges:
        origin_city = int(edge[0])
        dest_city = int(edge[1])
        if not g.has_edge(origin_city, dest_city):
            g.add_edge(origin_city, dest_city, weight=0)
        g.get_edge_data(origin_city, dest_city)['weight'] += edge[2]
    return g


dataset = 'ucsocial'
dic = {'ploblogs': 1224, 'ucsocial': 1899, 'condmat': 16264, 'wiki': 90153}
nodes_nums = dic[dataset]

for ith in range(1):#here we just create one result document for example
    edges = pd.read_csv('data/' + dataset + '/' + dataset + '_numpy' + str(ith) + '.csv')
    edges.columns = ['source', 'target', 'weight']
    g = read_graph(edges.values, nodes_nums)
    print('graph '+str(g.number_of_nodes())+', '+str(g.number_of_edges()))

    test_edges = pd.read_csv('data/' + dataset + '/' + dataset + 'sam' + str(ith) + '.csv').values
    for testedge in test_edges:
        # use "-" as flag for testedges
        if not g.has_edge(int(testedge[0]), int(testedge[1])):
            g.add_edge(int(testedge[0]), int(testedge[1]), weight=0)
        g.get_edge_data(int(testedge[0]), int(testedge[1]))['weight'] += -testedge[2]

    # degree table
    deg_list = {}
    for node in g.nodes:
        deg = len(g.adj[node]) + (node in g.adj[node])
        deg_list[node] = deg

    edges = edges.iloc[:, :-1].values
    # edges = edges[:, :-1].astype(np.int)
    edges = pd.DataFrame(edges)
    edges = edges[edges[0] != edges[1]].values
    # candidate set
    edges_set = edges.tolist()
    edges_set = [tuple(x) for x in edges_set]

    e = edges_set[0]

    edges_set = set(edges_set)
    i = 0
    for layer in range(1):#here we just compress once for example

        while len(edges_set) > 0:
            min_deg_edge = 100000
            for edge_item in edges_set:
                edge_deg = deg_list[edge_item[0]] + deg_list[edge_item[1]]  # edge_degree
                if edge_deg < min_deg_edge:
                    min_deg_edge = edge_deg
                    e = edge_item
            # print('e='+str(e)+', deg='+str(min_deg_edge))
            e0, e1 = int(e[0]), int(e[1])

            common_neighbors = list(nx.common_neighbors(g, e0, e1))
            # check whether has "-" or not
            has_test_edge = 0
            for cn in common_neighbors:
                if g.get_edge_data(e0, cn)['weight'] < 0 or g.get_edge_data(e1, cn)['weight'] < 0:
                    has_test_edge = 1
                    break
            if has_test_edge == 1:
                edges_set.remove(e)
                continue
            # no "-", then compress
            Ni = list(nx.neighbors(g, e0))
            Nj = list(nx.neighbors(g, e1))
            if len(Ni) > len(Nj):  
                for n in Nj:
                    w = g.get_edge_data(e1, n)['weight']
                    if not g.has_edge(e0, n):
                        g.add_edge(e0, n, weight=0)

                        # 更新度表
                        deg_list[e0] = deg_list[e0] + 1
                        deg_list[n] = deg_list[n] + 1     # 不能抵消
                    g.get_edge_data(e0, n)['weight'] += w
                    deg_list[n] = deg_list[n] - 1   # 合并在一个循环内

                g.remove_node(e1)
                del deg_list[e1]

                edges_set.remove(e)

                oriset = edges_set
                subset = set()
                subset1 = set()

                for ite in oriset:
                    if (ite[0] != e1) and (ite[1] != e1) and (ite[0] != e0) and (ite[1] != e0):
                        subset1.add(ite)
                # oriset = oriset - subset
                # edges_set = oriset
                del oriset
                edges_set = subset1
            else:  # 把Ni连接到j节点上,删除i节点
                for n in Ni:
                    w = g.get_edge_data(e0, n)['weight']
                    if not g.has_edge(e1, n):
                        g.add_edge(e1, n, weight=0)

                        deg_list[e1] = deg_list[e1] + 1
                        deg_list[n] = deg_list[n] + 1  # 不能抵消

                    g.get_edge_data(e1, n)['weight'] += w
                    deg_list[n] = deg_list[n] - 1  # 合并在一个循环内

                g.remove_node(e0)
                del deg_list[e0]
                edges_set.remove(e)

                oriset = edges_set
                subset = set()
                subset1 = set()
                for ite in oriset:
                    if (ite[0] != e1) and (ite[1] != e1) and (ite[0] != e0) and (ite[1] != e0):
                        subset1.add(ite)
                # oriset = oriset - subset
                # edges_set = oriset
                del oriset
                edges_set = subset1

            i = i + 1
            print('compress '+str(i))
        print('g.info = '+str(g.number_of_nodes())+', '+str(g.number_of_edges()))

        new_edge_table = nx.to_pandas_edgelist(g)
        edges = new_edge_table[new_edge_table['weight'] > 0]
        edges = edges.iloc[:, :-1]
        edges = edges[edges['source'] != edges['target']].values
        # next layer's candidate set
        edges_set = edges.tolist()
        edges_set = [tuple(x) for x in edges_set]
        edges_set = set(edges_set)
        print(str(ith) + str(layer) + ' complete!')

    print('g.info = ' + str(g.number_of_nodes()) + ', ' + str(g.number_of_edges()))
    new_edge_table = nx.to_pandas_edgelist(g)
    aset = set(new_edge_table['source']) | set(new_edge_table['target'])
    nodes = len(aset)
    print('nodes = ', nodes)

    node_dic = {}
    for i, n in enumerate(aset):
        node_dic[n] = i
    new_edge_table['source'] = new_edge_table['source'].map(node_dic)
    new_edge_table['target'] = new_edge_table['target'].map(node_dic)
    # nodes = new_edge_table.values.max() + 1
    nodes = pd.DataFrame([nodes])
    nodes.to_csv('data/' + dataset + '/' + 'compress/' + dataset + '_allcompress_dis_deg_layer1' + str(ith), sep='\t', index=False, header=False)

    new_train = new_edge_table[new_edge_table['weight'] > 0]
    new_test = new_edge_table[new_edge_table['weight'] < 0]
    new_test['weight'] = -new_test['weight']
    new_train.to_csv('data/' + dataset + '/' + 'compress/' + dataset + '_allcompress_dis_deg_layer1' + str(ith), sep='\t', index=False, header=False, mode='a')
    new_test.to_csv('data/' + dataset + '/' + 'compress/' + dataset + '_allcompress_sam_dis_deg_layer1' + str(ith)+'.csv', index=False)
    print(str(ith) + ' complete!')
