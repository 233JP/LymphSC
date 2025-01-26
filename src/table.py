# -*- coding: utf-8 -*-
"""
Author:Jzh
Ddte:2024年12月18日--22:31
不要停止奔跑
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import Lipinski


# # 示例 SMILES 字符串
# smiles_list = ['C(C1=CC(C2C=CC=CC=2)=C(C(F)(F)F)S1)1ON=C(C2C=CC=C(C=2)C(F)(F)F)N=1',
#                'C(C1=CC(C2=C(C=CC=C2)C(F)(F)F)=CS1)1ON=C(C2C=C(C=CC=2)C(F)(F)F)N=1',
#                'C(C1=CC(C2C=CC(CC)=CC=2)=CS1)1ON=C(C2C=C(C=CC=2)C(F)(F)F)N=1',
#                'CC1=NC(C2=CC(C(F)(F)F)=CC=C2)=NO1',
#                'FC(C1=CC=CC(C2=NOC(C3=CC(C4=CC=CC=C4)=CS3)=N2)=C1)(F)F',
#                'C1(C2=CC=CC=C2)=CSC=C1']
#
# # 模型预测结果，假设模型已经预测出来了
# predictions = [1, 1, 1, 0, 1, 0]  # 1为能溶，0为不能溶
#
# # 存储数据的列表
# data = []
#
# # 提取每个 SMILES 的特征
# for i, smiles in enumerate(smiles_list):
#     mol = Chem.MolFromSmiles(smiles)
#
#     if mol is None:
#         continue  # 如果无法解析SMILES，跳过
#
#     # 提取化学特征
#     mol_weight = Descriptors.MolWt(mol)  # 分子量
#     logp = Descriptors.MolLogP(mol)  # LogP
#     num_h_acceptors = Lipinski.NumHAcceptors(mol)  # 氢键受体数
#     psa = Descriptors.TPSA(mol)  # 极性表面积
#     num_rot_bonds = Descriptors.NumRotatableBonds(mol)  # 可旋转键数
#     num_rings = rdMolDescriptors.CalcNumRings(mol)  # 环数量
#     num_h_donors = Lipinski.NumHDonors(mol)  # 氢键供体数
#
#     # 将结果放入数据列表
#     data.append(
#         [smiles, predictions[i], mol_weight, logp, psa, num_h_acceptors, num_rot_bonds, num_rings, num_h_donors])
#
# # 创建 DataFrame
# df = pd.DataFrame(data, columns=["SMILES", "Prediction", "Molecular Weight", "LogP",
#                                  "Polar Surface Area", "H-bond Acceptors", "Rotatable Bonds",
#                                  "Ring Count", "H-bond Donors"])
#
# # 显示表格
# print(df)
#
# # 将表格保存为 CSV 文件
# df.to_csv('drug_predictions_with_selected_features.csv', index=False)

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import math
from rdkit.Chem import AllChem

# 计算分子级别的属性
def get_molecular_features(mol):
    num_atoms = mol.GetNumAtoms()  # 原子数量
    num_bonds = mol.GetNumBonds()  # 键数量
    has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())  # 是否有芳香性
    has_ring = any(atom.IsInRing() for atom in mol.GetAtoms())  # 是否有环
    molecular_weight = Descriptors.ExactMolWt(mol)  # 分子量
    log_p = Descriptors.MolLogP(mol)  # logP 值
    return num_atoms, num_bonds, has_aromatic, has_ring, molecular_weight, log_p

# 计算原子层面的统计信息
def get_atomic_stats(mol):
    # 计算 Gasteiger 电荷
    AllChem.ComputeGasteigerCharges(mol)
    charges = [
        atom.GetDoubleProp('_GasteigerCharge')
        for atom in mol.GetAtoms()
        if not (math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(atom.GetDoubleProp('_GasteigerCharge')))
    ]
    avg_charge = sum(charges) / len(charges) if charges else 0  # Gasteiger电荷均值
    total_hydrogen = sum(atom.GetTotalNumHs(includeNeighbors=True) for atom in mol.GetAtoms())
    avg_hydrogen_per_atom = total_hydrogen / mol.GetNumAtoms()
    return avg_charge, avg_hydrogen_per_atom

# SMILES 字符串统计
def get_smiles_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    num_atoms, num_bonds, has_aromatic, has_ring, molecular_weight, log_p = get_molecular_features(mol)
    avg_charge, avg_hydrogen_per_atom = get_atomic_stats(mol)
    smiles_length = len(smiles)  # SMILES 字符串长度
    return {
        "SMILES": smiles,
        "原子数量": num_atoms,
        "键数量": num_bonds,
        "是否含芳香环": has_aromatic,
        "分子量": molecular_weight,
        "logP": log_p,  # 新增的 logP 值
        "Gasteiger 电荷均值": avg_charge,
        "平均氢原子数": avg_hydrogen_per_atom,
        "SMILES 长度": smiles_length,
    }

# 示例 SMILES 和预测结果
smiles_list = ['C(C1=CC(C2C=CC=CC=2)=C(C(F)(F)F)S1)1ON=C(C2C=CC=C(C=2)C(F)(F)F)N=1',
                'C(C1=CC(C2=C(C=CC=C2)C(F)(F)F)=CS1)1ON=C(C2C=C(C=CC=2)C(F)(F)F)N=1',
                'C(C1=CC(C2C=CC(CC)=CC=2)=CS1)1ON=C(C2C=C(C=CC=2)C(F)(F)F)N=1',
                'CC1=NC(C2=CC(C(F)(F)F)=CC=C2)=NO1',
                'FC(C1=CC=CC(C2=NOC(C3=CC(C4=CC=CC=C4)=CS3)=N2)=C1)(F)F',
                'C1(C2=CC=CC=C2)=CSC=C1']  # 新的 SMILES
predictions = [1, 1, 1, 0, 1, 0]  # 模型预测结果
confidences = [0.95, 0.89]  # 预测置信度

# 构造表格
data = []
for smiles, pred in zip(smiles_list, predictions):
    features = get_smiles_features(smiles)
    features["预测结果"] = "溶解" if pred == 1 else "不溶解"
    data.append(features)

# 转为 Pandas DataFrame 并保存
df = pd.DataFrame(data)
print(df)
df.to_csv("results_table.csv", index=False)  # 保存为 CSV 文件
