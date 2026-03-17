"""
目的：用90分位线展示观测不完备时，观测数值的抖动？与之对应的是，完备上报观测时，观测数据与真实数值的趋势完全一致。
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.ticker as mtick
import os

from train_gpu import my_test, _config_args
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
from pyproj import CRS, Geod

config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 16,
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)

light_blue = '#ADD8E6'    # Light Blue
sky_blue = '#87CEEB'      # Sky Blue
baby_blue = '#89CFF0'     # Baby Blue

args = _config_args()
args.env_data_dir = '../../data/'
env = EpidemicModel(args, env_count=1)
OD = env.OD[0]
POP = env.POP[0].cpu().numpy()

def _plot_three_level_compare(
    true_series_2d: np.ndarray,
    perfect_obs_2d: np.ndarray,
    partial_obs_2d: np.ndarray,
    title: str,
    save_path: str,
):
    """
    输入：
        - 三个二维数组，形状 (num_seeds, time_steps)
    功能：
        - 对每个数组沿种子维做均值与 10/90 分位
        - 绘制均值曲线 + 10–90 分位带
    """
    def _band_stats(arr):
        mean = np.nanmean(arr, axis=0)
        p10  = np.nanpercentile(arr, 10, axis=0)
        p90  = np.nanpercentile(arr, 90, axis=0)
        return mean, p10, p90

    true_mean, true_p10, true_p90 = _band_stats(true_series_2d)
    perf_mean, perf_p10, perf_p90 = _band_stats(perfect_obs_2d)
    part_mean, part_p10, part_p90 = _band_stats(partial_obs_2d)

    t_len = true_mean.shape[0]
    x = np.arange(t_len)

    fig, ax = plt.subplots(figsize=(7, 5))

    # 真实状态：黑色
    ax.fill_between(x, true_p10, true_p90, color="blue", alpha=0.12)
    ax.plot(x, true_mean, label="Fully observable", linewidth=2, color="blue")

    # 完备上报：天蓝
    ax.fill_between(x, perf_p10, perf_p90, color="orange", alpha=0.25)
    ax.plot(x, perf_mean, label="Steady partially obs.", linewidth=2.0, color="orange")

    # 部分上报：浅蓝（或婴儿蓝）
    ax.fill_between(x, part_p10, part_p90, color=light_blue, alpha=0.25)
    ax.plot(x, part_mean, label="Non-steady partially obs.", linewidth=2.0, color=light_blue)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$I_{\mathbf{combined}}$ (persons)")
    ax.grid(True, linestyle="--", alpha=0.4)
    legend = ax.legend(frameon=False)
    ax.margins(x=0)
    # plt.tight_layout()

    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_observation_levels_compare():
    # 读取数据，计算三个观测等级下的观测数据
    simRes = np.load("data/simRes1.npy")
    perfect_obs = np.load("data/perfect_obs.npy")
    partial_obs = np.load("data/partial_obs.npy")

    true_state = simRes[:, :, :, [env.E_undetected, env.E_detected,
                                  env.I_undetected, env.I_detected, env.I_reported]].sum(axis=-1) # (env_count, period, zone_num)
    perfect_obs = perfect_obs[:, :, :, 0]
    partial_obs = partial_obs[:, :, :, 0]

    # 绘制整体的
    true_state_overall = true_state.sum(axis=-1)
    perfect_obs_overall = perfect_obs.sum(axis=-1)
    partial_obs_overall = partial_obs.sum(axis=-1)
    _plot_three_level_compare(true_state_overall, perfect_obs_overall, partial_obs_overall,
                              title="Overall", save_path="figure/overall_en.png")
    # 绘制某一个区域的
    for idx in [359, 363]:
        _plot_three_level_compare(true_state[:, :, idx], perfect_obs[:, :, idx], partial_obs[:, :, idx],
                              title=f"Region {idx}", save_path=f"figure/region_{idx}_en.png")

    # for idx in range(0, 500, 20):
    #     _plot_three_level_compare(true_state[:, :, idx], perfect_obs[:, :, idx], partial_obs[:, :, idx],
    #                               title=f"Region {idx}", save_path=f"figure/region_{idx}.png")


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
        # cmap='Reds',
        cmap='Oranges',
        scheme='UserDefined',
        classification_kwds={'bins': bins_eps},
        legend=show_legend,
        ax=ax,
        edgecolor='darkgrey',
        linewidth=0.5
    )

    # 绘制比例尺
    if show_legend:
        add_scalebar(ax, gdf, length_km=20, where=(0.35, 0.03), fontsize=16)
        # add_north(ax)

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

def output_community_geo3(true_state, perfect_obs, partial_obs, color_grade, output_path):
    """绘制 完全可观测-完备上报观测-部分可观测 数据"""
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

    # 2.完备上报观测
    perfect_obs_csv = mapping.copy()
    perfect_obs_csv['value'] = perfect_obs[perfect_obs_csv["filtered_idx"].values]
    new_gdf = gdf.merge(perfect_obs_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(axs[1], new_gdf, color_grade, legend_title=legend_title)

    # 2.部分可观测
    partial_obs_csv = mapping.copy()
    partial_obs_csv['value'] = partial_obs[partial_obs_csv["filtered_idx"].values]
    new_gdf = gdf.merge(partial_obs_csv, left_on='OBJECTID', right_on='OBJECTID')
    plot_community_geo_ax(axs[2], new_gdf, color_grade, legend_title=legend_title)

    fig.subplots_adjust(left=0.0, right=0.99, top=0.99, bottom=0.0, wspace=0.0, hspace=0.0)
    plt.savefig(output_path+'/total_compare_true_perfect_partial.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_community_geo_ax(ax, new_gdf, color_grade, show_legend=True)
    plt.tight_layout()
    plt.savefig(output_path + '/total_compare_true_perfect_partial_lengend.png', dpi=300)
    plt.show()

    rmse1 = cal_rmse(perfect_obs, true_state)
    rmse2 = cal_rmse(partial_obs, true_state)
    print(f"rmse1={rmse1}, rmse2={rmse2}")


def cal_rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

def plot_observation_compare_in_geo():
    # 读取数据，计算三个观测等级下的观测数据
    simRes = np.load("data/simRes1.npy")
    daily_new_E = np.load("data/daily_new_E.npy")
    perfect_obs = np.load("data/perfect_obs.npy")
    partial_obs = np.load("data/partial_obs.npy")



    # 1. 现存感染者
    true_state = simRes[:, :, :, [env.E_undetected, env.E_detected,
                                  env.I_undetected, env.I_detected, env.I_reported]].sum(
        axis=-1)  # (env_count, period, zone_num)

    DAY_IDX = 21
    true_state = true_state[0, DAY_IDX, :]
    perfect_observation = perfect_obs[0, DAY_IDX, :, 0]
    partial_observation = partial_obs[0, DAY_IDX, :, 0]
    # color_grade = [0, 20, 50, 100, 200, 500, 1500]
    color_grade = [0, 20, 50, 150, 500, 1500]
    output_path = "figure/curr_I"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # output_community_geo3(true_state, perfect_observation, partial_observation, color_grade, output_path)

    # 2. 新增暴露者
    true_state = daily_new_E  # (env_count, period, zone_num)

    DAY_IDX = 21
    true_state = true_state[0, DAY_IDX, :]
    perfect_observation = perfect_obs[0, DAY_IDX, :, 1]
    partial_observation = partial_obs[0, DAY_IDX, :, 1]
    color_grade = [0, 20, 50, 100, 200, 500]
    output_path = "figure/new_E"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_community_geo3(true_state, perfect_observation, partial_observation, color_grade, output_path)


from pyproj import CRS
from pyproj.geod import Geod


def add_north(ax, x=0.05, y=0.9, size=0.06):
    ax.annotate(
        'N',
        xy=(x, y + size),
        xytext=(x, y),
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center', va='center',
        arrowprops=dict(arrowstyle='-|>', lw=1.2, color='k', mutation_scale=15)
    )

def add_scalebar(ax, gdf, length_km=20, where=(0.35, 0.04),
                 tick_fracs=(0, 0.5, 1.0), tick_labels=None,
                 unit="km", unit_on_last_only=True,
                 linewidth=2, fontsize=9, color="k"):
    """
    在地图上添加比例尺

    参数:
    - ax: matplotlib坐标轴对象
    - gdf: GeoDataFrame对象
    - length_km: 比例尺长度（公里）
    - where: 比例尺位置，相对于地图范围的相对坐标 (x, y)
    - tick_fracs: 刻度位置（比例值，0到1之间）
    - tick_labels: 刻度标签，如果为None则自动生成
    - unit: 单位
    - unit_on_last_only: 是否只在最后一个刻度显示单位
    - linewidth: 比例尺线宽
    - fontsize: 字体大小
    - color: 颜色
    """

    # 不要覆盖传入的 tick_fracs 参数
    # tick_fracs = (0, 4/8, 1.0)  # 删除这行

    # 自动生成刻度标签
    if tick_labels is None:
        labs = []
        for i, f in enumerate(tick_fracs):
            v = length_km * f

            # 处理标签格式
            if unit_on_last_only and i != len(tick_fracs) - 1:
                # 不是最后一个刻度，不显示单位
                if f == 0:
                    labs.append("0")
                elif f == 0.5:
                    labs.append(f"{v:.0f}")  # 0.5位置显示10
                else:
                    labs.append(f"{v:.0f}")
            else:
                # 最后一个刻度，显示单位
                if f == 0:
                    labs.append("0")
                else:
                    labs.append(f"{v:.0f} {unit}")
        tick_labels = labs

    # 获取地图边界
    minx, miny, maxx, maxy = gdf.total_bounds

    # 计算比例尺起点位置
    x0 = minx + where[0] * (maxx - minx)
    y0 = miny + where[1] * (maxy - miny)

    # 计算刻度高度和文本偏移（使用更合理的相对值）
    dy = (maxy - miny)
    # 使用更稳定的计算方法
    tick_h = dy * 0.03  # 刻度高度为地图高度的0.5%
    text_off = dy * 0.045  # 文本偏移为地图高度的1.5%

    # 获取坐标系信息
    crs = CRS.from_user_input(gdf.crs) if gdf.crs else None

    if crs is not None and crs.is_projected:
        # 投影坐标系：可以直接使用米
        # 确保坐标系单位是米
        if crs.axis_info[0].unit_name == 'metre':
            L = length_km * 1000.0  # 公里转米
        else:
            # 如果单位不是米，需要转换
            # 这里简单假设单位是米，实际情况可能需要更复杂的处理
            L = length_km * 1000.0

        # 绘制比例尺主线
        x1 = x0 + L
        ax.plot([x0, x1], [y0, y0], color=color, lw=linewidth, zorder=10)

        # 绘制刻度和标签
        for frac, lab in zip(tick_fracs, tick_labels):
            xx = x0 + L * float(frac)
            # 绘制刻度线
            ax.plot([xx, xx], [y0 - tick_h, y0 + tick_h], color=color, lw=1.5, zorder=10)
            # 添加刻度标签
            ax.text(xx, y0 - text_off, lab,
                    ha="center", va="top",
                    fontsize=fontsize, color=color, zorder=10)

    else:
        # 地理坐标系（经纬度）：需要进行测地计算
        geod = Geod(ellps="WGS84")
        lon0, lat0 = float(x0), float(y0)

        # 计算比例尺终点（向东方向）
        lon1, lat1, _ = geod.fwd(lon0, lat0, 90, length_km * 1000.0)

        # 绘制比例尺主线
        ax.plot([lon0, lon1], [lat0, lat0], color=color, lw=linewidth, zorder=10)

        # 绘制刻度和标签
        for frac, lab in zip(tick_fracs, tick_labels):
            # 计算每个刻度点的位置
            loni, lati, _ = geod.fwd(lon0, lat0, 90, length_km * 1000.0 * float(frac))
            # 绘制刻度线
            ax.plot([loni, loni], [lat0 - tick_h, lat0 + tick_h], color=color, lw=1.5, zorder=10)
            # 添加刻度标签
            ax.text(loni, lat0 - text_off, lab,
                    ha="center", va="top",
                    fontsize=fontsize, color=color, zorder=10)

    # 添加比例尺标题（可选）
    # ax.text(x0, y0 + 2 * tick_h, "比例尺",
    #        ha="left", va="bottom",
    #        fontsize=fontsize, color=color, zorder=10)

    return ax

if __name__ == '__main__':
    plot_observation_levels_compare()
    # plot_observation_compare_in_geo()