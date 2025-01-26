# -*- coding: utf-8 -*-
"""
Author:Jzh
Ddte:2025年01月03日--14:51
不要停止奔跑
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import torch_geometric.nn as pyg_nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        # 增加层数和隐藏单元数量
        self.conv1 = GATConv(in_channels, 16, heads=4, dropout=0.6)  # 第一层 GAT
        self.conv2 = GATConv(16*4, 32, heads=4, dropout=0.6)         # 第二层 GAT
        self.conv3 = GATConv(32*4, out_channels, heads=1, concat=False, dropout=0.6)  # 第三层 GAT, 输出层

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # 加入 batch 信息
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)

        # 使用全局池化将节点信息聚合成图的表示
        x = pyg_nn.global_mean_pool(x, batch)  # 对每个图进行汇聚，得到图的表示

        return torch.sigmoid(x)