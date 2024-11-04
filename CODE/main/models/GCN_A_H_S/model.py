# cython: language_level=3
# -*-coding: utf-8 -*-
# @Time : 2022/3/20 15:46
# @Author : 
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.GCN_A_H_S.graph import Graph
from models.GCN_A_H_S.layers import *


class GCNE(nn.Module):
    def __init__(self, in_dim, n_classes, training=True, dropout=0.0, device="cuda:0"):
        super(GCNE, self).__init__()

        self.device = device
        self.training = training
        self.dropout = dropout
        self.n_classes = n_classes
        attention_dim = 5
        head_number = 5

        self.edge_transformer_1_1 = Transformer(attention_dim, 5, 4,dropout=self.dropout)
        self.edge_transformer_1_2 = Transformer(attention_dim, 5, 4,dropout=self.dropout)
        self.edge_transformer_2_1 = Transformer(attention_dim, 5, 4,dropout=self.dropout)
        self.edge_transformer_2_2 = Transformer(attention_dim, 5, 4,dropout=self.dropout)
        self.edge_transformer_3_1 = Transformer(attention_dim, 5, 4,dropout=self.dropout)
        self.edge_transformer_3_2 = Transformer(attention_dim, 5, 4,dropout=self.dropout)

        self.node_graphConv_1 = MultiHeadGCNLayer(in_dim, 130, head_number,dropout=self.dropout)
        self.node_graphConv_2 = MultiHeadGCNLayer(130, 130, head_number,dropout=self.dropout)
        self.node_graphConv_3 = MultiHeadGCNLayer(130, 130, head_number,dropout=self.dropout)

        self.linear_1 = nn.Linear(130*3, 130)
        self.linear_2 = nn.Linear(130, 64)
        self.linear_3 = nn.Linear(64, n_classes)
        self.dropout1 = nn.Dropout(self.dropout)
        self.output = nn.LogSoftmax(dim=1)

    def forward_(self, g, mode):
        adj_node = g.node_adj()
        i = torch.eye(adj_node.shape[0]).to(self.device)
        features = g.node_feature
        edge = g.edge_feature

        edge_1 = self.edge_transformer_1_1(edge)
        edge_2 = self.edge_transformer_1_2(edge)
        edge_f_1 = F.tanh(edge_1)
        edge_f_2 = F.tanh(edge_2)
        edge_f = g.edge_feature_trans(edge_f_1,edge_f_2)+i

        h1 = F.leaky_relu(self.node_graphConv_1(edge_f, features))

        edge_1 = (self.edge_transformer_2_1(edge_1) + edge_1)
        edge_2 = (self.edge_transformer_2_2(edge_2) + edge_2)
        edge_f_1 = F.tanh(edge_1)
        edge_f_2 = F.tanh(edge_2)
        edge_f = g.edge_feature_trans(edge_f_1,edge_f_2)+i

        h2 = F.leaky_relu(self.node_graphConv_2(edge_f, h1))

        edge_1 = (self.edge_transformer_3_1(edge_1) + edge_1)
        edge_2 = (self.edge_transformer_3_2(edge_2) + edge_2)
        edge_f_1 = F.tanh(edge_1)
        edge_f_2 = F.tanh(edge_2)
        edge_f = g.edge_feature_trans(edge_f_1,edge_f_2)+i

        h3 = F.leaky_relu(self.node_graphConv_3(edge_f, h2))

        if mode == 'sub':
            h1 = torch.mean(h1, dim=0).view(1, -1)
            h2 = torch.mean(h2, dim=0).view(1, -1)
            h3 = torch.mean(h3, dim=0).view(1, -1)
        elif mode == 'batch':
            h1 = Graph.aggregation(g, h1)
            h2 = Graph.aggregation(g, h2)
            h3 = Graph.aggregation(g, h3)
        else:
            raise ValueError('mode error')

        h1 = torch.cat([h1, h2, h3], dim=1)
        h1 = F.leaky_relu(self.linear_1(h1))
        h1 = F.leaky_relu(self.linear_2(h1))
        h1 = self.linear_3(h1)
        h1 = self.dropout1(h1)
        return self.output(h1)

    def forward(self, gs):
        if isinstance(gs, Graph):
            return self.forward_(gs, mode='batch')
        else:
            return torch.cat([self.forward_(g, mode='sub') for g in gs], dim=0)

def loss_function(DEVICE):
    return nn.NLLLoss().to(DEVICE)

def get_model(config_):
    return GCNE(config_.NODE_FEATURE_NUM, config_.CLASSIFY_NUM)

def draw_attention_hot(data,name):
    import matplotlib.pyplot as plt
    import numpy as np
    data = data.cpu().detach().numpy()
    for d in range(data.shape[0]):
        p = rf'/to/path/CODE/main/reprocess/atten_hot/{name}_{d}.csv'
        np.savetxt(p,data[d,:,:],delimiter=',')
