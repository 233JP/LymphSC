import json
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import math

# 读取 JSON 数据
# with open('./../Data/Data/lymSC.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

with open('augmented_smiles_a.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# 提取 SMILES 和输出标签
smiles_list = [entry['SMILES of Chemical structures'] for entry in data]
labels_list = [entry['output'] for entry in data]


scaler_node = StandardScaler()
scaler_edge = StandardScaler()

# 生成one-hot编码的函数
def one_hot_vector(value, valid_values, add_unknown=True):
    """
    为给定的值生成一个one-hot编码的向量。
    """
    vec = np.zeros(len(valid_values))
    if value in valid_values:
        vec[valid_values.index(value)] = 1
    elif add_unknown:
        vec[-1] = 1  # 如果值不在valid_values中，添加未知类别
    return vec

def get_atom_features(atom):
    """
    为给定原子生成特征，返回一个长度为 d_atom 的向量。
    """
    # 100+1=101维
    v1 = one_hot_vector(atom.GetAtomicNum(), [i for i in range(1, 101)])

    # 5+1=6维
    v2 = one_hot_vector(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])

    # 8维：原子的其他特征
    v3 = [
        atom.GetTotalNumHs(includeNeighbors=True) / 8,
        atom.GetDegree() / 4,
        atom.GetFormalCharge() / 8,
        atom.GetTotalValence() / 8,
        # 检查 '_GasteigerCharge' 是否存在
        0 if not atom.HasProp('_GasteigerCharge') or math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerCharge'))
        else atom.GetDoubleProp('_GasteigerCharge'),

        # 检查 '_GasteigerHCharge' 是否存在
        0 if not atom.HasProp('_GasteigerHCharge') or math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerHCharge'))
        else atom.GetDoubleProp('_GasteigerHCharge'),

        int(atom.GetIsAromatic()),
        int(atom.IsInRing())
    ]

    # 原子位置编码
    v4 = [atom.GetIdx() + 1]  # 从1开始

    # 合并特征
    attributes = np.concatenate([v1, v2, v3, v4], axis=0)

    # 返回115维的特征
    # assert len(attributes) == d_atom
    return attributes

def get_bond_features(bond):
    """
    为给定的化学键生成特征，返回一个长度为 d_edge 的向量。
    """
    # 4维：键类型
    v1 = one_hot_vector(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                             Chem.rdchem.BondType.DOUBLE,
                                             Chem.rdchem.BondType.TRIPLE,
                                             Chem.rdchem.BondType.AROMATIC], add_unknown=False)

    # 6维：键的立体化学
    v2 = one_hot_vector(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOANY,
                                           Chem.rdchem.BondStereo.STEREOCIS,
                                           Chem.rdchem.BondStereo.STEREOE,
                                           Chem.rdchem.BondStereo.STEREONONE,
                                           Chem.rdchem.BondStereo.STEREOTRANS,
                                           Chem.rdchem.BondStereo.STEREOZ], add_unknown=False)

    # 3维：键的其他特征
    v3 = [
        int(bond.GetIsConjugated()),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing())
    ]

    # 合并所有特征
    attributes = np.concatenate([v1, v2, v3])

    # 返回128维的特征
    # assert len(attributes) == d_edge
    return attributes

def smiles_to_graph(smiles):
    """
    将 SMILES 字符串转换为图数据。
    """
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    # 创建邻接矩阵和边特征矩阵
    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()

        adj_matrix[i, j] = bond_type
        adj_matrix[j, i] = bond_type

        # 计算边特征
        bond_attributes = get_bond_features(bond)
        edge_features.append(bond_attributes)
        edge_features.append(bond_attributes)  # 由于是无向图，每条边需要出现两次

    # 创建节点特征矩阵
    feature_matrix = []
    for atom in mol.GetAtoms():
        atom_attributes = get_atom_features(atom)
        feature_matrix.append(atom_attributes)

    feature_matrix = np.array(feature_matrix)
    edge_features = np.array(edge_features)

    # 归一化
    scaler_node = StandardScaler()
    scaler_edge = StandardScaler()
    feature_matrix = scaler_node.fit_transform(feature_matrix)
    edge_features = scaler_edge.fit_transform(edge_features)

    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
    x = torch.tensor(feature_matrix, dtype=torch.float)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


data_list = []
labels = []

for smiles, label in zip(smiles_list, labels_list):
    graph_data = smiles_to_graph(smiles)
    data_list.append(graph_data)
    labels.append(label)

# 将标签转换为 PyTorch 张量
labels = torch.tensor(labels, dtype=torch.long)

# 划分训练集和测试集
# train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_data, test_data, train_labels, test_labels = train_test_split(data_list, labels, test_size=0.2, random_state=42)

train_dataset = [(graph, label) for graph, label in zip(train_data, train_labels)]
test_dataset = [(graph, label) for graph, label in zip(test_data, test_labels)]
train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)


# 定义 GCN 模型
class GCNNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, num_layers=3, dropout=0.5):
        super(GCNNet, self).__init__()
        self.num_layers = num_layers

        # 第一层卷积
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))

        # 中间层卷积
        for _ in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 最后一层卷积
        self.convs.append(GCNConv(hidden_dim, out_channels))

        # BatchNorm 和 Dropout
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.num_layers - 1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 应用多层 GCNConv + BatchNorm + Dropout
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)  # Batch Normalization
            x = torch.relu(x)
            x = self.dropout(x)  # Dropout

        # 最后一层卷积，不需要 BatchNorm 和 Dropout
        x = self.convs[-1](x, edge_index)

        # 使用全局池化将节点信息聚合成图的表示
        x = pyg_nn.global_mean_pool(x, batch)  # 对每个图进行汇聚，得到图的表示

        return torch.sigmoid(x)

# model = GCNNet(in_channels=train_data[0].x.shape[1], out_channels=1)  # 二分类任务，输出维度为1
model = GCNNet(in_channels=train_data[0].x.shape[1], out_channels=1, hidden_dim=64, num_layers=4, dropout=0.5)
initial_lr = 0.003
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
loss_fn = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

train_losses = []
test_accuracies = []

# 训练模型
for epoch in range(30):
    model.train()
    total_loss = 0
    for batch in train_loader:
        graphs, labels = batch
        optimizer.zero_grad()
        outputs = model(graphs)

        loss = loss_fn(outputs.squeeze(), labels.float())  # 使用 batch.y 作为标签
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)  # 保存每个epoch的平均损失
    current_lr = optimizer.param_groups[0]['lr']

    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Learning Rate:{current_lr:.6f}')

    # 更新学习率
    scheduler.step()

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    labels_1 = []
    outputs_1 = []
    with torch.no_grad():
        for batch in test_loader:
            graphs, labels = batch
            outputs = model(graphs)
            predicted = (outputs > 0.5).float().squeeze()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            labels_1.append(labels.detach().numpy())
            outputs_1.append(outputs.squeeze().detach().numpy())
    labels_concatenated = np.concatenate(labels_1)
    outputs_1_concatenated = np.concatenate(outputs_1)
    auc = roc_auc_score(labels_concatenated, outputs_1_concatenated)
    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)  # 保存每个epoch的测试准确率
    print(f'AUC-ROC:{auc}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
# # 测试模型
# model.eval()
# correct = 0
# total = 0
# for batch in test_loader:
#     outputs = model(batch)
#     predicted = (outputs > 0.5).float()
#     predicted = predicted.squeeze()
#     correct += (predicted == batch.y).sum().item()
#     total += batch.y.size(0)
# print(f'correct:{correct}')
# print(f'total:{total}')
# print(f'Test Accuracy: {correct/total:.4f}')

# 可视化 Loss 和 Test Accuracy
epochs = range(1, 31)

# 绘制损失图
plt.figure(figsize=(12, 5))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# 绘制测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Over Epochs')
plt.legend()


# 显示图表
plt.tight_layout()
plt.show()

print(train_losses)
print(test_accuracies)