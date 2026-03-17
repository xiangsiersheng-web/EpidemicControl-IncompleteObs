import numpy as np
import pandas as pd

# 生成一个 n*2 的随机连续数据集 (n=10, 2列代表两个目标)
np.random.seed(42)  # 保证结果可重复
n = 10000
data = np.array([
    np.random.normal(loc=50, scale=5, size=n),  # 目标1的数据集中在40-50之间
    np.random.uniform(10, 100, n)  # 目标2的数据分布在10-100之间，分散程度大
]).T

# 将数据转为DataFrame格式
df = pd.DataFrame(data, columns=['目标1', '目标2'])

# 数据归一化（Min-Max归一化）
df_normalized = (df - df.min()) / (df.max() - df.min())

# 计算每个目标的p_ij值（每个样本的归一化值占比）
p = df_normalized.div(df_normalized.sum(axis=0), axis=1)

# 计算熵值 e_j
k = 1 / np.log(n)  # 归一化系数
e_j = -k * (p * np.log(p + 1e-9)).sum(axis=0)  # 加1e-9是为了避免log(0)的问题

# 计算权重 w_j
w_j = (1 - e_j) / (2 - e_j.sum())

# 返回结果
print(df, '\n', df_normalized, '\n', e_j, '\n', w_j)
