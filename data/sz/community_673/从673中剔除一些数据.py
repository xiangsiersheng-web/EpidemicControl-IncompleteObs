"""
673 的数据中，有些区域的人口过少，或者流动异常，要剔除
"""
import numpy as np
import pandas as pd
import os

OD = np.load("flow.npy")
POP = np.load("population.npy")


# 1. 人口少于 1000 的剔除
valid_population_idx = np.where(POP >= 1000)[0]

# 2. OD 中对角元素大于 0.8 的剔除
valid_od_idx = np.where(np.diag(OD) <= 0.8)[0]

# 3. 取两者的交集
valid_idx = np.intersect1d(valid_population_idx, valid_od_idx)
print("原始区域数量:", len(POP), "剩余区域数量:", len(valid_idx))

# 过滤后的数据
filtered_OD = OD[valid_idx][:, valid_idx]
filtered_POP = POP[valid_idx]
filtered_OD = filtered_OD / filtered_OD.sum(axis=-1, keepdims=True) # 归一化

# 记录映射关系
original_to_filtered = {original: new for new, original in enumerate(valid_idx)}
filtered_to_original = {new: original for original, new in original_to_filtered.items()}


output_path = f"../community_{len(valid_idx)}"
if not os.path.exists(output_path):
    os.mkdir(output_path)
print("output path:", output_path)


np.save(output_path + "/flow.npy", filtered_OD)
np.save(output_path + "/population.npy", filtered_POP)

# 保存映射关系到文件
mapping_df = pd.DataFrame({
    "original_idx": list(original_to_filtered.keys()),
    "filtered_idx": list(original_to_filtered.values()),
    "OBJECTID": [key + 1 for key in original_to_filtered.keys()]
})
mapping_df.to_csv(output_path + "/mapping.csv", index=False)

