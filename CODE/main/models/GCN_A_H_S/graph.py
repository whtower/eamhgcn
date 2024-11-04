# cython: language_level=3
# -*-coding: utf-8 -*-
# @Time : 2022/4/12 13:06
# @Author : 

import torch


class Graph:
    def __init__(self, graph_index, node_feature, edge_feature,file_path=None):
        if 'torch.LongTensor' == type(graph_index):
            self.graph_index = graph_index
        else:
            self.graph_index = torch.LongTensor(graph_index)

        if 'torch.FloatTensor' == type(node_feature):
            self.node_feature = node_feature
        else:
            self.node_feature = torch.FloatTensor(node_feature)

        if 'torch.FloatTensor' == type(edge_feature):
            self.edge_feature = edge_feature
        else:
            self.edge_feature = torch.FloatTensor(edge_feature)

        self.device = "cpu"
        self.node_index = None
        self.edge_index = None
        self.adj = None
        self.file_path = file_path

    @staticmethod
    def batch(g):
        node_index = [0]
        edge_index = [0]
        graph_index = []
        node_feature = []
        edge_feature = []
        all_node_index = 0
        all_edge_index = 0
        file_paths = []
        for i in g:
            i_graph_index = torch.add(i.graph_index, torch.tensor([all_node_index,all_node_index,all_edge_index]))
            node_feature.append(i.node_feature) # [:,[0,1,2,3,4,5]]
            edge_feature.append(i.edge_feature)
            i__node_sum,i_edge_sum = i.__len__()
            all_node_index += i__node_sum
            all_edge_index += i_edge_sum
            node_index.append(all_node_index)
            edge_index.append(all_edge_index)
            graph_index.append(i_graph_index)
            file_paths.append(i.file_path)
        graph_index = torch.cat(graph_index, dim=0)
        node_feature = torch.cat(node_feature, dim=0)
        edge_feature = torch.cat(edge_feature, dim=0)
        graph = Graph(graph_index, node_feature, edge_feature,file_paths)
        graph.add_index(node_index, edge_index)
        return graph

    def __len__(self):
        return self.node_feature.shape[0], self.edge_feature.shape[0]

    def to(self, device):
        self.device = device
        if self.edge_feature is not None:
            self.edge_feature = self.edge_feature.to(self.device)
        if self.node_feature is not None:
            self.node_feature = self.node_feature.to(self.device)
        return self

    def add_index(self, node_index, edge_index):
        if 'torch.LongTensor' == type(node_index):
            self.node_index = node_index
        else:
            self.node_index = torch.LongTensor(node_index)
        if 'torch.LongTensor' == type(edge_index):
            self.edge_index = edge_index
        else:
            self.edge_index = torch.LongTensor(edge_index)

    def node_adj(self):
        node_num = self.__len__()[0]
        self.adj = torch.zeros((node_num, node_num))
        self.adj[self.graph_index[:,0], self.graph_index[:,1]] = 1
        self.adj[self.graph_index[:,1], self.graph_index[:,0]] = 1 # 无向图
        return self.adj

    def edge_feature_trans(self,f1,f2):
        node_num,feat_ch = self.__len__()[0],f1.shape[1]
        edge_feature = torch.zeros((feat_ch, node_num, node_num)).to(self.device)
        edge_feature[:, self.graph_index[:,0], self.graph_index[:,1]] = f1[self.graph_index[:,2],:].T #上三角
        edge_feature[:, self.graph_index[:,1], self.graph_index[:,0]] = f2[self.graph_index[:,2],:].T #下三角
        return edge_feature

    @staticmethod
    def aggregation(g, h, method="max"):
        node_index = g.node_index
        if method == "max":
            return torch.stack([(h[node_index[i]:node_index[i + 1], :]).max(0)[0] for i in range(node_index.__len__() - 1)])
        elif method == "mean":
            return torch.stack([torch.mean(h[node_index[i]:node_index[i + 1], :], dim=0) for i in range(node_index.__len__() - 1)])
        elif method == "sum":
            return torch.stack([torch.sum(h[node_index[i]:node_index[i + 1], :], dim=0) for i in range(node_index.__len__() - 1)])
        else:
            raise Exception("No such method")
