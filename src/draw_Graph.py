# -*- coding: utf-8 -*-
"""
Author:Jzh
Ddte:2024年11月14日--16:24
不要停止奔跑
"""
import argparse
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from dataset_graph import construct_dataset, mol_collate_func
from transformer_graph_finetune import make_model
from utils import ScheduledOptim, get_options, get_loss, cal_loss, evaluate, scaffold_split
from collections import defaultdict
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdchem
import dgl
import networkx as nx


import math
import torch


from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

def one_hot_vector(val, lst, add_unknown=True):
    if add_unknown:
        vec = np.zeros(len(lst) + 1)
    else:
        vec = np.zeros(len(lst))

    vec[lst.index(val) if val in lst else -1] = 1
    return vec


def get_atom_features(atom, d_atom):
    # 100+1=101 dimensions
    v1 = one_hot_vector(atom.GetAtomicNum(), [i for i in range(1, 101)])

    # 5+1=6 dimensions
    v2 = one_hot_vector(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                  Chem.rdchem.HybridizationType.SP2,
                                                  Chem.rdchem.HybridizationType.SP3,
                                                  Chem.rdchem.HybridizationType.SP3D,
                                                  Chem.rdchem.HybridizationType.SP3D2])

    # 8 dimensions
    v3 = [
        atom.GetTotalNumHs(includeNeighbors=True) / 8,
        atom.GetDegree() / 4,
        atom.GetFormalCharge() / 8,
        atom.GetTotalValence() / 8,
        # 0 if math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(
        #     atom.GetDoubleProp('_GasteigerCharge')) else atom.GetDoubleProp('_GasteigerCharge'),
        # 0 if math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(
        #     atom.GetDoubleProp('_GasteigerHCharge')) else atom.GetDoubleProp('_GasteigerHCharge'),
        0 if not atom.HasProp('_GasteigerCharge') or math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerCharge')) else atom.GetDoubleProp('_GasteigerCharge'),
        0 if not atom.HasProp('_GasteigerHCharge') or math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerHCharge')) else atom.GetDoubleProp('_GasteigerHCharge'),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing())
    ]

    # index for position encoding
    v4 = [
        atom.GetIdx() + 1  # start from 1
    ]

    attributes = np.concatenate([v1, v2, v3, v4], axis=0)

    # total for 32 dimensions
    assert len(attributes) == d_atom + 1
    return attributes


def get_bond_features(bond, d_edge):
    # 4 dimensions
    v1 = one_hot_vector(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                             Chem.rdchem.BondType.DOUBLE,
                                             Chem.rdchem.BondType.TRIPLE,
                                             Chem.rdchem.BondType.AROMATIC], add_unknown=False)

    # 6 dimensions
    v2 = one_hot_vector(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOANY,
                                           Chem.rdchem.BondStereo.STEREOCIS,
                                           Chem.rdchem.BondStereo.STEREOE,
                                           Chem.rdchem.BondStereo.STEREONONE,
                                           Chem.rdchem.BondStereo.STEREOTRANS,
                                           Chem.rdchem.BondStereo.STEREOZ], add_unknown=False)

    # 3 dimensions
    v3 = [
        int(bond.GetIsConjugated()),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing())
    ]

    # total for 115+13=128 dimensions
    attributes = np.concatenate([v1, v2, v3])

    assert len(attributes) == d_edge
    return attributes


def load_data_from_mol(mol, d_atom, d_edge, max_length):
    # Set Stereochemistry
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)
    AllChem.ComputeGasteigerCharges(mol)

    # Get Node features Init
    node_features = np.array([get_atom_features(atom, d_atom) for atom in mol.GetAtoms()])

    # Get Bond features
    num_atoms = mol.GetNumAtoms()
    bond_features = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), d_edge))

    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx = bond.GetEndAtom().GetIdx()
        bond_features[begin_atom_idx, end_atom_idx, :] = bond_features[end_atom_idx, begin_atom_idx, :] = get_bond_features(bond, d_edge)

    # Get Adjacency matrix without self loop
    adjacency_matrix = Chem.rdmolops.GetDistanceMatrix(mol).astype(float)
    return adjacency_matrix
    plt.figure(figsize=(len(adjacency_matrix[0]), len(adjacency_matrix[0])))
    plt.imshow(adjacency_matrix, cmap='Blues', interpolation='none')
    plt.colorbar(label='Connection Strength')
    plt.title('Adjacency Matrix Visualization')
    plt.xlabel('Atomic Weight')
    plt.ylabel('Atomic Weight')
    plt.show()

    # node_features.shape    = (num_atoms, d_atom) -> (max_length, d_atom)
    # bond_features.shape    = (num_atoms, num_atoms, d_edge) -> (max_length, max_length, d_edge)
    # adjacency_matrix.shape = (num_atoms, num_atoms) -> (max_length, max_length)
    # return pad_array(node_features, (max_length, node_features.shape[-1])), \
    #        pad_array(bond_features, (max_length, max_length, bond_features.shape[-1])), \
    #        pad_array(adjacency_matrix, (max_length, max_length))


class Molecule:

    def __init__(self, mol, label, d_atom, d_edge, max_length):
        self.smile = Chem.MolToSmiles(mol)
        self.label = label
        self.node_features, self.bond_features, self.adjacency_matrix = load_data_from_mol(mol, d_atom, d_edge, max_length)


class MolDataSet(Dataset):

    def __init__(self, data_list):
        self.data_list = np.array(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataSet(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape):
    padded_array = np.zeros(shape, dtype=float)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array


def construct_dataset(mol_list, label_list, d_atom, d_edge, max_length):
    output = [Molecule(mol, label, d_atom, d_edge, max_length)
              for (mol, label) in tqdm(zip(mol_list, label_list), total=len(mol_list))]

    return MolDataSet(output)


def mol_collate_func(batch):
    smile_list, adjacent_list, node_feature_list, bond_feature_list, label_list = [], [], [], [], []

    for molecule in batch:
        smile_list.append(molecule.smile)
        adjacent_list.append(molecule.adjacency_matrix)
        node_feature_list.append(molecule.node_features)
        bond_feature_list.append(molecule.bond_features)

        if isinstance(molecule.label, list):       # task number != 1
            label_list.append(molecule.label)
        else:                                      # task number == 1
            label_list.append([molecule.label])

    return [smile_list] + [torch.from_numpy(np.array(features)).float() for features in (adjacent_list, node_feature_list, bond_feature_list, label_list)]


def construct_loader(mol_list, label_list, batch_size, d_atom, d_edge, max_length, shuffle=True):
    dataset = construct_dataset(mol_list, label_list, d_atom, d_edge, max_length)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=mol_collate_func, shuffle=shuffle,
                        drop_last=True, num_workers=0)
    return loader


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help='choose a dataset', default='lymph')
    parser.add_argument("--split", type=str, help="choose the split type", default='random',
                        choices=['random', 'scaffold', 'cv'])

    args = parser.parse_args()

    # load options
    model_params, train_params = get_options(args.dataset)

    # load data
    if train_params['task'] == 'regression':
        with open(f'./../Data/Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
            [data_mol, data_label, data_mean, data_std] = pkl.load(f)
    else:
        with open(f'./../Data/Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
            [data_mol, data_label] = pkl.load(f)

    # calculate the padding
    # model_params['max_length'] = max([data.GetNumAtoms() for data in data_mol])
    # print(f"Max padding length is: {model_params['max_length']}")
    #
    # dataset = construct_dataset(data_mol, data_label, model_params['d_atom'], model_params['d_edge'], model_params['max_length'])
    # for i in data_mol:
        # if len(i) < 40:
            # print(i.shape)
    # a = len(data_mol[39])
    # print(data_mol[39])
    mol = data_mol[39]
    num = 1
    # 遍历分子中的每个原子，提取属性
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        features = get_atom_features(atom, 115)
        print(f'num:{num}, atom:{symbol}, features:{features}')
        num += 1


    # mol = Chem.MolFromSmiles(smiles)
    # adj = load_data_from_mol(data_mol[39], 115, 13, 96)
    # print(adj.shape)
    # # 设置全局字体大小
    # plt.rcParams.update({'font.size': 16})  # 16是字体大小，可根据需求调整
    # # 可视化邻接矩阵
    # plt.figure(figsize=(7, 7))  # 调整图像大小
    # plt.imshow(adj, cmap='Blues', interpolation='none')
    # # 单独设置标题和标签的字体大小
    # plt.title('Adjacency Matrix Visualization', fontsize=18)  # 标题字体大小
    # plt.xlabel('Atomic Weight', fontsize=16)  # x轴标签字体大小
    # plt.ylabel('Atomic Weight', fontsize=16)  # y轴标签字体大小
    # # 添加色条并设置色条标签的字体大小
    # cbar = plt.colorbar(label='Connection Strength')
    # cbar.ax.tick_params(labelsize=12)  # 色条刻度字体大小
    # cbar.set_label('Connection Strength', fontsize=16)  # 色条标签字体大小\
    # plt.savefig('adj_matrix')
    # plt.show()
# ---------------------------------------------------------------------------------------------

    # node_features = np.random.rand(20, 116)  # 使用随机数生成特征矩阵示例
    # # 可视化节点特征矩阵
    # plt.figure(figsize=(10, 8))  # 调整图像大小
    # plt.imshow(node_features, cmap='viridis', aspect='auto', interpolation='none')
    # # 添加色条来指示特征值强度
    # cbar = plt.colorbar(label='Feature Value')
    # cbar.ax.tick_params(labelsize=14)
    # cbar.set_label('Feature Value', fontsize=16)
    # # 设置标题和坐标轴标签
    # plt.title('Node Feature Matrix Visualization', fontsize=20)
    # plt.xlabel('Feature Index (116 Features)', fontsize=16)
    # plt.ylabel('Node Index (20 Nodes)', fontsize=16)
    # plt.show()

    # 使用RDKit解析SMILES
    # smiles = "CN1C2=C(C=C(Cl)C=C2)C(=NCC1=O)C1=CC=CC=C1"  # 这里输入您的药物的SMILES字符串
    # mol = Chem.MolFromSmiles(smiles)
    #
    # # 初始化属性列表
    # atom_types = []
    # valence_electrons = []
    # electronegativity = []
    # hybridization_values = []
    #
    # # 遍历分子中的每个原子，提取属性
    # for atom in mol.GetAtoms():
    #     atom_types.append(atom.GetSymbol())
    #     valence_electrons.append(atom.GetTotalValence())
    #
    #     # 使用周期表的电负性值（根据原子类型）
    #     electronegativity_dict = {'H': 2.1, 'C': 2.5, 'N': 3.0, 'O': 3.5}  # 示例电负性值，可根据需要扩展
    #     electronegativity.append(electronegativity_dict.get(atom.GetSymbol(), 0))
    #
    #     # 获取杂化态并转换为数值
    #     hybridization = atom.GetHybridization()
    #     hybridization_map = {rdchem.HybridizationType.SP: 1,
    #                          rdchem.HybridizationType.SP2: 2,
    #                          rdchem.HybridizationType.SP3: 3}
    #     hybridization_values.append(hybridization_map.get(hybridization, 0))
    #
    # # 将不同的特征绘制在同一张散点图上
    # # plt.rcParams.update({'font.size': 16})  # 16是字体大小，可根据需求调整
    # plt.figure(figsize=(10, 6))
    #
    # # 原子类型散点
    # atom_type_set = list(set(atom_types))
    # atom_type_indices = [atom_type_set.index(atom) for atom in atom_types]
    # plt.scatter(range(len(atom_types)), atom_type_indices, color='purple', label='Atom Type', s=50)
    #
    # # 价电子数散点
    # plt.scatter(range(len(valence_electrons)), valence_electrons, color='blue', label='Valence Electrons', s=50)
    #
    # # 电负性散点
    # plt.scatter(range(len(electronegativity)), electronegativity, color='green', label='Electronegativity', s=50)
    #
    # # 杂化态散点
    # plt.scatter(range(len(hybridization_values)), hybridization_values, color='red',
    #             label='Hybridization (1=sp, 2=sp2, 3=sp3)', s=50)
    #
    # # 设置图例和标题
    # plt.xlabel('Atom Index',fontsize=16)
    # plt.ylabel('Value',fontsize=16)
    # plt.title('Distribution of Atomic Properties',fontsize=18)
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()

# 1. 解析SMILES，生成RDKit分子对象
# smiles = 'CN1N(C(=O)C=C1C)C1=CC=CC=C1'  # 可以替换为任何其他SMILES
# mol = Chem.MolFromSmiles(smiles)
# 3. 创建DGL图，设置节点和边特征
# def create_dgl_graph(mol, d_atom, d_edge):
#     G = dgl.DGLGraph()
#
#     # 添加节点（原子）
#     num_atoms = mol.GetNumAtoms()
#     G.add_nodes(num_atoms)
#
#     # 添加边（化学键）
#     bonds = mol.GetBonds()
#     bond_indices = []
#     for bond in bonds:
#         start_idx = bond.GetBeginAtomIdx()
#         end_idx = bond.GetEndAtomIdx()
#         G.add_edge(start_idx, end_idx)
#         bond_indices.append((start_idx, end_idx))
#
#     # 计算节点特征
#     node_features = np.array([get_atom_features(atom, d_atom) for atom in mol.GetAtoms()])
#     G.ndata['features'] = torch.tensor(node_features, dtype=torch.float32)
#
#     # 计算边特征
#     bond_features = np.array([get_bond_features(bond, d_edge) for bond in bonds])
#     edge_features = []
#     for (start_idx, end_idx) in bond_indices:
#         edge_features.append(bond_features[bonds.index(mol.GetBondBetweenAtoms(start_idx, end_idx))])
#
#     G.edata['features'] = torch.tensor(edge_features, dtype=torch.float32)
#
#     return G
# def create_dgl_graph(mol, d_atom, d_edge):
#     # 获取原子特征
#     node_features = np.array([get_atom_features(atom, d_atom) for atom in mol.GetAtoms()])
#
#     # 获取键特征
#     bond_features = []
#     bonds = []
#
#     # 遍历所有键并获取特征
#     for bond in mol.GetBonds():
#         bond_features.append(get_bond_features(bond, d_edge))
#         bonds.append(bond)
#
#     # 初始化图
#     G = dgl.DGLGraph()
#
#     # 添加节点
#     G.add_nodes(len(mol.GetAtoms()))
#     G.ndata['features'] = torch.tensor(node_features, dtype=torch.float32)
#
#     # 添加边
#     edge_indices = []
#     for bond in bonds:
#         start_idx = bond.GetBeginAtomIdx()
#         end_idx = bond.GetEndAtomIdx()
#         edge_indices.append((start_idx, end_idx))
#
#     # 将边加入图中
#     G.add_edges([x[0] for x in edge_indices], [x[1] for x in edge_indices])
#
#     # 将边特征加入图中
#     G.edata['features'] = torch.tensor(bond_features, dtype=torch.float32)
#
#     return G
#
# # 4. 创建DGL图
# d_atom = 115  # 根据你的需求设置节点特征维度
# d_edge = 13  # 根据你的需求设置边特征维度
# G = create_dgl_graph(mol, d_atom, d_edge)
#
#
# # 5. 可视化图（使用Matplotlib和NetworkX）
# def draw_dgl_graph(G):
#     nx_graph = G.to_networkx().to_undirected()
#     pos = nx.spring_layout(nx_graph)  # 图的布局
#     plt.figure(figsize=(8, 8))
#
#     # 节点标签和边的标签（仅显示节点编号）
#     nx.draw(nx_graph, pos, with_labels=True, node_size=1500, node_color='skyblue', font_size=20, font_weight='bold',
#             edge_color='black', width=8)
#
#     plt.show()
#
#
# # 6. 可视化DGL图
# draw_dgl_graph(G)