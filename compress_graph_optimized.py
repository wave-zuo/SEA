import pandas as pd
import numpy as np
import networkx as nx
import random
import time
import heapq


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


dataset = 'condmat'
dic = {'neural': 296, 'celegans': 453, 'netscience': 575, 'ploblogs': 1224, 'ussocial': 1899, 'condmat': 16264,
       'wiki': 90153}
nodes_nums = dic[dataset]

# ith = 0
for ith in range(1):
    edges = pd.read_csv('data/' + dataset + '/' + dataset + '_numpy' + str(ith) + '.csv')
    edges.columns = ['source', 'target', 'weight']
    g = read_graph(edges.values, nodes_nums)
    print('graph ' + str(g.number_of_nodes()) + ', ' + str(g.number_of_edges()))

    test_edges = pd.read_csv('data/' + dataset + '/' + dataset + 'sam' + str(ith) + '.csv').values
    for testedge in test_edges:
        # -1表示是测试集，不能合并
        if not g.has_edge(int(testedge[0]), int(testedge[1])):
            g.add_edge(int(testedge[0]), int(testedge[1]), weight=0)
        g.get_edge_data(int(testedge[0]), int(testedge[1]))['weight'] += -testedge[2]

    # degree表
    deg_list = {}
    for node in g.nodes():
        deg = len(g.adj[node]) + (node in g.adj[node])
        deg_list[node] = deg

    edges = edges.iloc[:, :-1].values
    # edges = edges[:, :-1].astype(np.int)
    edges = pd.DataFrame(edges)
    edges = edges[edges[0] != edges[1]].values
    # 候选集
    edges_set = edges.tolist()
    # heapq要求首个字段为优先级
    edges_set = [(deg_list[x[0]] + deg_list[x[1]], tuple(x)) for x in edges_set]
    heapq.heapify(edges_set)  # 建堆
    print(edges_set)

    e = edges_set[0][1]
    # quick表为真正候选集，堆中不能修改元素，存在历史数据，需利用quick表辅助判断
    quick_edgeset = {}
    for (edge_deg, e) in edges_set:
        quick_edgeset[e] = edge_deg
    i = 0
    start_time = time.time()
    # 仅压缩一轮进行示例
    while len(quick_edgeset) > 0:
        e = edges_set[0][1]    # 堆顶元素
        if e not in quick_edgeset:
            heapq.heappop(edges_set)    # 不存在或历史元素
            continue

        e0, e1 = int(e[0]), int(e[1])

        common_neighbors = list(nx.common_neighbors(g, e0, e1))
        has_test_edge = 0
        for cn in common_neighbors:
            if g.get_edge_data(e0, cn)['weight'] < 0:
                has_test_edge = 1
                if (cn, e1) in quick_edgeset:
                    del quick_edgeset[(cn, e1)]
                elif (e1, cn) in quick_edgeset:
                    del quick_edgeset[(e1, cn)]
                break
            elif g.get_edge_data(e1, cn)['weight'] < 0:
                has_test_edge = 1
                if (cn, e0) in quick_edgeset:
                    del quick_edgeset[(cn, e0)]
                elif (e0, cn) in quick_edgeset:
                    del quick_edgeset[(e0, cn)]
                break
            # if g.get_edge_data(e0, cn)['weight'] < 0 or g.get_edge_data(e1, cn)['weight'] < 0:
            #     has_test_edge = 1
            #     break

        if has_test_edge == 1:
            del quick_edgeset[e]
            continue
        # 安全时才能合并
        Ni = list(nx.neighbors(g, e0))
        Nj = list(nx.neighbors(g, e1))
        if len(Ni) > len(Nj):  # 把Nj连接到i节点上,删除j节点
            for n in Nj:
                w = g.get_edge_data(e1, n)['weight']
                if not g.has_edge(e0, n):
                    g.add_edge(e0, n, weight=w)
                    # 更新度表
                    deg_list[e0] = deg_list[e0] + 1
                    # n的度先增加1，由于后续要删除j,抵消
                else:  # 存在，增加权重
                    g.get_edge_data(e0, n)['weight'] += w
                    deg_list[n] = deg_list[n] - 1  # 删除j造成的

            g.remove_node(e1)
            del deg_list[e1]

            del quick_edgeset[e]

            for en in Ni:
                if (e0, en) in quick_edgeset:
                    del quick_edgeset[(e0, en)]
                elif (en, e0) in quick_edgeset:
                    del quick_edgeset[(en, e0)]
            for en in Nj:
                if (e1, en) in quick_edgeset:
                    del quick_edgeset[(e1, en)]
                elif (en, e1) in quick_edgeset:
                    del quick_edgeset[(en, e1)]
            for n in common_neighbors:      # 共同邻居的边度才会减小
                for x in nx.neighbors(g, n):
                    if (n, x) in quick_edgeset:
                        quick_edgeset[(n, x)] -= 1
                        # 堆中元素不能修改，插入新优先级元组，该元组一定在原元组之前被取到，
                        # 取到后若可以操作则在117行从quick表中删除，以防后续历史数据影响
                        heapq.heappush(edges_set, (quick_edgeset[(n, x)], (n, x)))

                    elif (x, n) in quick_edgeset:
                        quick_edgeset[(x, n)] -= 1
                        heapq.heappush(edges_set, (quick_edgeset[(x, n)], (x, n)))

        else:  # 把Ni连接到j节点上,删除i节点
            for n in Ni:
                w = g.get_edge_data(e0, n)['weight']
                if not g.has_edge(e1, n):
                    g.add_edge(e1, n, weight=w)
                    # 更新度表
                    deg_list[e1] = deg_list[e1] + 1
                else:  # 存在，增加权重
                    g.get_edge_data(e1, n)['weight'] += w
                    deg_list[n] = deg_list[n] - 1  # 删除j造成的

            g.remove_node(e0)
            del deg_list[e0]
            del quick_edgeset[e]

            for en in Ni:
                if (e0, en) in quick_edgeset:
                    del quick_edgeset[(e0, en)]
                elif (en, e0) in quick_edgeset:
                    del quick_edgeset[(en, e0)]
            for en in Nj:
                if (e1, en) in quick_edgeset:
                    del quick_edgeset[(e1, en)]
                elif (en, e1) in quick_edgeset:
                    del quick_edgeset[(en, e1)]
            for n in common_neighbors:
                for x in nx.neighbors(g, n):
                    if (n, x) in quick_edgeset:
                        quick_edgeset[(n, x)] -= 1
                        heapq.heappush(edges_set, (quick_edgeset[(n, x)], (n, x)))

                    elif (x, n) in quick_edgeset:
                        quick_edgeset[(x, n)] -= 1
                        heapq.heappush(edges_set, (quick_edgeset[(x, n)], (x, n)))

        i = i + 1
        # print('compress '+str(i))
    print('g.info = ' + str(g.number_of_nodes()) + ', ' + str(g.number_of_edges()))
    end_time = time.time()
    print('compress time:', end_time - start_time)

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
    nodes.to_csv('data/' + dataset + '/' + 'compress/' + dataset + '_allcompress_dis_deg_layer1' + str(ith), sep='\t',
                 index=False, header=False)

    new_train = new_edge_table[new_edge_table['weight'] > 0]
    new_test = new_edge_table[new_edge_table['weight'] < 0]
    new_test['weight'] = -new_test['weight']
    new_train.to_csv('data/' + dataset + '/' + 'compress/' + dataset + '_allcompress_dis_deg_layer1' + str(ith),
                     sep='\t', index=False, header=False, mode='a')
    new_test.to_csv(
        'data/' + dataset + '/' + 'compress/' + dataset + '_allcompress_sam_dis_deg_layer1' + str(ith) + '.csv',
        index=False)
    print(str(ith) + ' complete!')
