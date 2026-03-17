"""
用于绘制感染曲线和动作，在地图上展示
"""
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from train_gpu import my_test, _config_args
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1


config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 16,
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)

### 地图以及映射关系加载
shp_file = "../data/sz/Shenzhen_geo_data/Shenzhen_Community.shp"
geodata = gpd.read_file(shp_file)
geodata.sort_values(by="OBJECTID", inplace=True)
print(len(geodata))

map_path = "../data/sz/community_654/mapping.csv"
map_objectid_idx = pd.read_csv(map_path)
print(len(map_objectid_idx))

### 要绘制数据的获取
with open('./感染和动作的地图绘制/env.pkl', 'rb') as f:  # 'rb' 表示以二进制读模式打开
    env = pickle.load(f)


def plot_geo_infections(geodata, map_objectid_idx, infections, title='', vmax=0.1) -> None:
    """
    绘制地理感染地图，根据感染数据填充区域颜色深浅。

    参数：
    geodata: GeoDataFrame，包含地图的地理数据和 OBJECTID。
    map_objectid_idx: DataFrame，包含 OBJECTID 和 newid 的映射关系。
    infections: numpy array，长度为 643，对应 new_id 的感染数据。
    title: str，绘图标题。
    """
    # 创建一个从 OBJECTID 到感染值的映射
    objectid_to_infections = pd.merge(
    geodata[['OBJECTID']],
    map_objectid_idx,
    on='OBJECTID',
    how='left'
    )

    # 将 newid 映射到 infections
    objectid_to_infections['infection'] = objectid_to_infections['filtered_idx'].apply(
    lambda x: infections[int(x)] if not pd.isna(x) else 0
    )

    # 将感染数据加入 geodata
    geodata['infection'] = objectid_to_infections['infection']

    # 绘制地图
    fig, ax = plt.subplots(figsize=(12, 10))
    plot = geodata.plot(
        column='infection',  # 根据 'infection' 列填充颜色
        # cmap='coolwarm',  # 颜色映射为冷暖色调
        # cmap='Reds',  # 颜色映射为冷暖色调
        cmap='Oranges',  # 颜色映射为冷暖色调
        legend=True,  # 显示图例
        legend_kwds={
            'label': "Test Rate",
            'shrink': 0.6,  # 调整颜色条大小
            'orientation': 'vertical',

        },
        ax=ax,  # 指定绘图的轴
        edgecolor='darkgrey',
        # edgecolor='lightgray',  # 区域边框颜色
        linewidth = 0.5,
        vmin=0,  # 手动设置颜色轴最小值
        vmax=vmax,  # 手动设置颜色轴最大值

    )

    # 如果需要科学计数法显示
    if True:
        # 获取颜色条
        cbar = plot.get_figure().axes[-1]
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 1))  # 设置指数范围
        cbar.yaxis.set_major_formatter(formatter)

    # 添加标题
    ax.set_title(title, fontsize=16)

    # 隐藏坐标轴
    ax.set_axis_off()

    # 显示绘图
    plt.show()

POP = np.load('../data/sz/community_654/population.npy')
# plot_geo_infections(geodata, map_objectid_idx, env.POP[0].cpu().numpy(), title='population')

for day in [0, 2, 4, 10]:
    infection = env.simRes[0, day, :, [1,2,3,4,5]].sum(dim=-1) / env.POP[0]
    infection = infection.cpu().numpy()
    plot_geo_infections(geodata, map_objectid_idx, infection, title=f'day {day}', vmax=1e-4)

    # obs = env.history_local_obs[0, day, :, 0].cpu().numpy()
    # plot_geo_infections(geodata, map_objectid_idx, obs, title=f'day {day}', vmax=1e-4)

    p_test, _ = env._action_to_u(env.actions[0, day, :])
    p_test = p_test.cpu().numpy()
    plot_geo_infections(geodata, map_objectid_idx, p_test, title=f'day {day}', vmax=0.05)


print(0)
