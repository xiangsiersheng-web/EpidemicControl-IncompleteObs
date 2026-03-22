"""
Among the 673 regions, some have too small population or abnormal flow, need to be removed
"""
import numpy as np
import pandas as pd
import os

OD = np.load("flow.npy")
POP = np.load("population.npy")


# 1. Remove regions with population less than 1000
valid_population_idx = np.where(POP >= 1000)[0]

# 2. Remove regions where diagonal elements of OD are greater than 0.8
valid_od_idx = np.where(np.diag(OD) <= 0.8)[0]

# 3. Take the intersection of both
valid_idx = np.intersect1d(valid_population_idx, valid_od_idx)
print("Original number of regions:", len(POP), "Remaining number of regions:", len(valid_idx))

# Filtered data
filtered_OD = OD[valid_idx][:, valid_idx]
filtered_POP = POP[valid_idx]
filtered_OD = filtered_OD / filtered_OD.sum(axis=-1, keepdims=True) # Normalize

# Record mapping relationship
original_to_filtered = {original: new for new, original in enumerate(valid_idx)}
filtered_to_original = {new: original for original, new in original_to_filtered.items()}


output_path = f"../community_{len(valid_idx)}"
if not os.path.exists(output_path):
    os.mkdir(output_path)
print("output path:", output_path)


np.save(output_path + "/flow.npy", filtered_OD)
np.save(output_path + "/population.npy", filtered_POP)

# Save mapping relationship to file
mapping_df = pd.DataFrame({
    "original_idx": list(original_to_filtered.keys()),
    "filtered_idx": list(original_to_filtered.values()),
    "OBJECTID": [key + 1 for key in original_to_filtered.keys()]
})
mapping_df.to_csv(output_path + "/mapping.csv", index=False)

