import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

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

args = _config_args()
args.env_data_dir = '../../data/'
env = EpidemicModel(args, env_count=1)
OD = env.OD[0]
POP = env.POP[0]

def plot_community_geo_ax(ax, gdf, color_grade, show_legend=False, legend_title=""):
    bins = [float(b) for b in color_grade]
    bins_eps = [b + 1e-10 for b in bins]

    # 生成图例标签
    # 采用左开右闭区间的描述，更贴合 mapclassify.UserDefined 的行为
    legend_labels = []
    legend_labels.append(f"≤ {int(bins[0]) if bins[0].is_integer() else bins[0]}")
    for i in range(1, len(bins)):
        lo = bins[i - 1]
        hi = bins[i]

        def _fmt(x):
            return int(x) if float(x).is_integer() else x

        legend_labels.append(f"({_fmt(lo)}, {_fmt(hi)}]")
    legend_labels.append(f"> {int(bins[-1]) if bins[-1].is_integer() else bins[-1]}")

    # 绘图
    plotted = gdf.plot(
        column='value',
        # cmap='RdYlGn_r',
        cmap='Reds',
        # cmap='Oranges',
        scheme='UserDefined',
        classification_kwds={'bins': bins_eps},
        legend=show_legend,
        ax=ax,
        edgecolor='darkgrey',
        linewidth=0.5
    )

    # 图例微调：位置、去边框、替换文字、方块标记
    leg = ax.get_legend()
    if leg is not None:
        # 1.2在1的基础上向右偏，1.1在1的基础上向上偏
        leg.set_bbox_to_anchor((1.3, 1), transform=ax.transAxes)
        leg.set_frame_on(False)
        # 替换文本
        for text, label in zip(leg.get_texts(), legend_labels):
            text.set_text(label)
        # 方块标记
        for h in leg.legendHandles:
            h.set_marker('s')
            h.set_markersize(10)

    # 视觉优化：去轴、去边框、等比
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('equal')

def plot_community_geo(true_state, partial_obs, rebuild_state, color_grade=[0, 1, 2, 3, 4, 5], title=''):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # 加载地图数据
    gdf = gpd.read_file("../../data/sz/Shenzhen_geo_data/Shenzhen_Community.shp")
    # 映射关系
    mapping = pd.read_csv("../../data/sz/community_654/mapping.csv")

    # 1.真实状态
    true_state_csv = mapping.copy()
    true_state_csv['value'] = true_state[true_state_csv["filtered_idx"].values]
    new_gdf = gdf.merge(true_state_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(axs[0], new_gdf, color_grade)

    # 2.部分可观测
    partial_obs_csv = mapping.copy()
    partial_obs_csv['value'] = partial_obs[partial_obs_csv["filtered_idx"].values]
    new_gdf = gdf.merge(partial_obs_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(axs[1], new_gdf, color_grade)

    # 3.重建状态
    rebuild_state_csv = mapping.copy()
    rebuild_state_csv['value'] = rebuild_state[rebuild_state_csv["filtered_idx"].values]
    new_gdf = gdf.merge(rebuild_state_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(axs[2], new_gdf, color_grade, show_legend=True)

    axs[0].set_title(f"True State (Day {day})")
    axs[1].set_title(f"Partial Observable (Day {day})")
    axs[2].set_title(f"Rebuild State (Day {day})")

    plt.show()


def output_community_geo(true_state, partial_obs, rebuild_state, color_grade, output_path):
    # 加载地图数据
    gdf = gpd.read_file("../../data/sz/Shenzhen_geo_data/Shenzhen_Community.shp")
    # 映射关系
    mapping = pd.read_csv("../../data/sz/community_654/mapping.csv")

    # 1.真实状态
    fig, ax = plt.subplots(figsize=(7, 4))
    true_state_csv = mapping.copy()
    true_state_csv['value'] = true_state[true_state_csv["filtered_idx"].values]
    new_gdf = gdf.merge(true_state_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(ax, new_gdf, color_grade)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.tight_layout()
    plt.savefig(output_path+'/true_state.png', dpi=300)
    plt.show()

    # 2.部分可观测
    fig, ax = plt.subplots(figsize=(7, 4))
    partial_obs_csv = mapping.copy()
    partial_obs_csv['value'] = partial_obs[partial_obs_csv["filtered_idx"].values]
    new_gdf = gdf.merge(partial_obs_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(ax, new_gdf, color_grade)
    plt.tight_layout()
    plt.savefig(output_path+'/partial_observable.png', dpi=300)
    plt.show()

    # 3.重建状态
    fig, ax = plt.subplots(figsize=(7, 4))
    rebuild_state_csv = mapping.copy()
    rebuild_state_csv['value'] = rebuild_state[rebuild_state_csv["filtered_idx"].values]
    new_gdf = gdf.merge(rebuild_state_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(ax, new_gdf, color_grade)
    plt.tight_layout()
    plt.savefig(output_path+'/rebuild_state.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_community_geo_ax(ax, new_gdf, color_grade, show_legend=True)
    plt.tight_layout()
    plt.savefig(output_path + '/rebuild_state2.png', dpi=300)
    plt.show()

    rmse1 = cal_rmse(partial_obs, true_state)
    rmse2 = cal_rmse(rebuild_state, true_state)
    print(f"rmse1={rmse1}, rmse2={rmse2}")


def output_community_geo2(true_state, partial_obs, rebuild_state, color_grade, output_path):
    # 加载地图数据
    gdf = gpd.read_file("../../data/sz/Shenzhen_geo_data/Shenzhen_Community.shp")
    # 映射关系
    mapping = pd.read_csv("../../data/sz/community_654/mapping.csv")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    legend_title = "Infections"
    if "new_E" in output_path:
        legend_title = "New Exposed"

    # 1.真实状态
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    true_state_csv = mapping.copy()
    true_state_csv['value'] = true_state[true_state_csv["filtered_idx"].values]
    new_gdf = gdf.merge(true_state_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(axs[0], new_gdf, color_grade, legend_title=legend_title)

    # 2.部分可观测
    partial_obs_csv = mapping.copy()
    partial_obs_csv['value'] = partial_obs[partial_obs_csv["filtered_idx"].values]
    new_gdf = gdf.merge(partial_obs_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(axs[1], new_gdf, color_grade, legend_title=legend_title)

    # 3.重建状态
    rebuild_state_csv = mapping.copy()
    rebuild_state_csv['value'] = rebuild_state[rebuild_state_csv["filtered_idx"].values]
    new_gdf = gdf.merge(rebuild_state_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(axs[2], new_gdf, color_grade, legend_title=legend_title)

    fig.subplots_adjust(left=0.0, right=0.99, top=0.99, bottom=0.0, wspace=0.0, hspace=0.0)
    plt.savefig(output_path+'/total_compare_true_partial_rebuild.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_community_geo_ax(ax, new_gdf, color_grade, show_legend=True)
    plt.tight_layout()
    plt.savefig(output_path + '/rebuild_state2.png', dpi=300)
    plt.show()

    rmse1 = cal_rmse(partial_obs, true_state)
    rmse2 = cal_rmse(rebuild_state, true_state)
    print(f"rmse1={rmse1}, rmse2={rmse2}")

def cal_rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

if __name__ == '__main__':

    simRes = np.load("partial_observable/simRes.npy")
    state1 = np.load("partial_observable/daily_new_E.npy")
    partial_obs = np.load("partial_observable/obs.npy")
    action1 = np.load("partial_observable/actions.npy")

    state2 = np.load("rebuild_info/daily_new_E.npy")
    rebuild2 = np.load("rebuild_info/rebuild_states.npy")
    action2 = np.load("rebuild_info/actions.npy")

    # state1/state2 在第二天应该是一样的
    DAY_INDEX = 7
    true_state1 = state1[0, DAY_INDEX, :]
    true_state2 = state2[0, DAY_INDEX, :]
    for i in range(true_state1.shape[0]):
        if true_state1[i] != true_state2[i]:
            print(f"region_idx={i}, state not equal")

    # # 新增E
    # for day in range(5, 10):
    #     true_state = state2[0, day, :]
    #     print(f"day={day}, true_state mean={true_state.mean()} max={true_state.max()}")
    #     partial_observe = obs1[0, day, :, 1]
    #     print(f"day={day}, partial_observe mean={partial_observe.mean()} max={partial_observe.max()}")
    #     rebuild_state = rebuild2[0, day, :, 1]
    #     print(f"day={day}, rebuild_state mean={rebuild_state.mean()} max={rebuild_state.max()}")
    #
    #     color_grade = [0, 1, 2, 3, 4, 5]
    #     plot_community_geo(true_state, partial_observe, rebuild_state, color_grade, title=f"new E(day={day})")

    # 选择day_idx=7展示
    day=7
    true_state = state2[0, day, :]
    partial_observe = partial_obs[0, day, :, 1]
    rebuild_state = rebuild2[0, day, :, 1]
    color_grade = [0, 1, 2, 3, 4]
    # output_community_geo(true_state, partial_observe, rebuild_state, color_grade, output_path='output_fig/new_E')
    output_community_geo2(true_state, partial_observe, rebuild_state, color_grade, output_path='output_fig/new_E')

    # # 现存I
    # print("------------------------------------------------------------------------------------------------")
    # for day in range(5, 10):
    #     true_state = simRes[0, day, :, :][:, [env.E_undetected, env.E_detected,
    #                                           env.I_undetected, env.I_detected, env.I_reported]].sum(axis=-1)
    #     print(f"day={day}, true_state mean={true_state.mean()} max={true_state.max()}")
    #     partial_observe = obs1[0, day, :, 0]
    #     print(f"day={day}, partial_observe mean={partial_observe.mean()} max={partial_observe.max()}")
    #     rebuild_state = rebuild2[0, day, :, 0]
    #     print(f"day={day}, rebuild_state mean={rebuild_state.mean()} max={rebuild_state.max()}")
    #
    #     color_grade = [0, 1, 2, 3, 4, 5]
    #     plot_community_geo(true_state, partial_observe, rebuild_state, color_grade, title=f"curr I(day={day})")

    # 选择day_idx=8展示
    day = 8
    true_state = simRes[0, day, :, :][:, [env.E_undetected, env.E_detected,
                                              env.I_undetected, env.I_detected, env.I_reported]].sum(axis=-1)
    partial_observe = partial_obs[0, day, :, 0]
    rebuild_state = rebuild2[0, day, :, 0]
    color_grade = [0, 1, 2, 3, 4, 5]
    output_community_geo2(true_state, partial_observe, rebuild_state, color_grade, output_path='output_fig/curr_I')

    pass
