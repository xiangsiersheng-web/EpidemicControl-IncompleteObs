import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString

# 1. 读取地图文件
shp_file = "Shenzhen_Community.shp"
data = gpd.read_file(shp_file)
data.sort_values(by="OBJECTID", inplace=True)
data.reset_index(drop=True, inplace=True)  # 重置索引以确保索引从0开始
num_areas = len(data)
print(f"社区数量: {num_areas}")

# 2. 生成流动矩阵
# 这里我们生成一个随机的流动矩阵作为示例，您可以根据实际数据替换
np.random.seed(42)  # 设置随机种子以确保结果可重复
flow_matrix = np.random.randint(0, 500, size=(num_areas, num_areas))
np.fill_diagonal(flow_matrix, 0)  # 设置对角线为0，表示自流动为0
print("流动矩阵示例：")
print(flow_matrix)

# 如果您有实际的流动数据，可以直接加载，例如从CSV文件或.npy文件
# 示例加载方法（请根据实际文件路径和格式调整）：
# flow_matrix = np.load('flow_matrix.npy')  # 从.npy文件加载
# 或
# flow_df = pd.read_csv("flow_matrix.csv", index_col=0)
# flow_matrix = flow_df.values

# 3. 设置过滤流量阈值
flow_threshold = 990  # 小于此流量的连接不绘制

# 4. 创建一个空的图（Graph）
G = nx.Graph()

# 将每个区域作为节点添加到图中
for idx, row in data.iterrows():
    region_id = idx  # 使用索引作为区域ID
    G.add_node(region_id, geometry=row.geometry)

# 5. 过滤并添加流动边
for i in range(num_areas):
    for j in range(i + 1, num_areas):  # 只遍历上三角矩阵，避免重复
        flow_value = flow_matrix[i, j] + flow_matrix[j, i]  # 双向流量之和

        if flow_value >= flow_threshold:
            # 在图中添加边，流量作为边的属性
            G.add_edge(i, j, weight=flow_value)
print("过滤后边数：", len(G.edges))
# 6. 绘制地图
fig, ax = plt.subplots(figsize=(12, 12))

# 绘制地图基础图层
data.plot(ax=ax, color='lightblue', edgecolor='black')

 # 绘制流动边
for u, v, edge_data in G.edges(data=True):
    # 获取区域的几何信息
    u_geom = G.nodes[u]["geometry"]
    v_geom = G.nodes[v]["geometry"]

    # 获取区域的代表点（质心）
    u_point = u_geom.centroid
    v_point = v_geom.centroid

    # 计算连接两区域的直线
    line = LineString([u_point, v_point])

    # 绘制边，线条的宽度与流量值成正比
    ax.plot(
        *line.xy,
        color="red",
        linewidth=edge_data["weight"] / 1000,  # 调整比例因子以适应实际流量范围
        alpha=0.6
    )

# 设置地图标题和其他参数
ax.set_title("人口流动地图", fontsize=15)
plt.axis('off')  # 关闭坐标轴

# 显示地图
plt.show()
