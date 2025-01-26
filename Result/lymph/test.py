# -*- coding: utf-8 -*-
"""
Author:Jzh
Ddte:2024年11月27日--17:17
不要停止奔跑
"""
import pandas as pd
from io import StringIO

data_io = 'best_test_result_None_linba_fold_1.csv'
df = pd.read_csv(data_io)


df['predict'] = df['predict'].str.replace('[', '').str.replace(']', '')

# 然后将字符串转换为浮点数
df['predict'] = df['predict'].astype(float)
df['actual'] = df['actual'].str.replace('[', '').str.replace(']', '')
df['actual'] = df['actual'].astype(float)

count_greater_than_zero = 0
count_less_than_zero = 0
# 遍历DataFrame的每一行
for index, row in df.iterrows():
    # 检查'predict'列的值是否大于0且'actual'列的值是否大于0
    if row['predict'] > 0 and row['actual'] > 0:
        count_greater_than_zero += 1
    # 检查'predict'列的值是否小于0且'actual'列的值是否等于0
    elif row['predict'] < 0 and row['actual'] == 0:
        count_less_than_zero += 1

# 打印结果
print(f"大于0的个数: {count_greater_than_zero}")
print(f"小于0的个数: {count_less_than_zero}")