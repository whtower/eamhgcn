# cython: language_level=3
# -*-coding: utf-8 -*-
# @Time : 2022/4/19 20:28
# @Author : 
import math
import torch
from torch import nn
from torch.nn import Parameter

# 多头GCN层
class MultiHeadGCNLayer(nn.Module):
    def __init__(self,in_features, out_features, num_heads=1,dropout=0.3):
        super(MultiHeadGCNLayer, self).__init__()
        assert out_features % num_heads == 0, "Please make sure the out_features can be divided by num_heads."
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.feature_len = out_features // num_heads
        self.linear = nn.Linear(in_features, out_features)
        self.weight = Parameter(torch.FloatTensor(num_heads, out_features, self.feature_len))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # 随机化参数
        self.bias.data.fill_(0)  # bias 初始化为0

    def forward(self, adj, feats):
        feats = self.linear(feats)
        mx = torch.matmul(feats, self.weight)
        mx = torch.matmul(adj, mx)
        mx = mx.permute((1,0,2)).contiguous()
        mx = mx.view((mx.shape[0], -1))
        mx = mx + self.bias
        mx = self.dropout(mx)
        mx = self.layer_norm(mx+feats)
        return mx


# 完整的Transfromer结构类
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.3, activation="relu"):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = self.transformer_encoder(src, mask, src_key_padding_mask)
        output = self.layer_norm(output+src)
        return output
