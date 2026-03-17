import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.ticker as mtick

from train_gpu import my_test, _config_args
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1

config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 # "font.size": 28, # 整体趋势-部分-重建
 # "font.size": 22, # 整体趋势-完备-部分
 "font.size": 25, # action_geo
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

light_blue = '#ADD8E6'    # Light Blue
sky_blue = '#87CEEB'      # Sky Blue
baby_blue = '#89CFF0'     # Baby Blue

args = _config_args()
args.env_data_dir = '../../data/'
env = EpidemicModel(args, env_count=1)
OD = env.OD[0]
POP = env.POP[0].cpu().numpy()
DAY = 30


def plot_overall_trend(simRes, obs, action, quara_num, output_path, is_rebuild=False, obs_curve_label='Partially Observable'):
    plt.rcParams.update(config)

    true_state = simRes[0, :, :, :][:, :, [env.E_undetected, env.E_detected,
                  env.I_undetected, env.I_detected, env.I_reported]].sum(axis=(1,2))

    partial_obs = obs[0, :, :, 0].sum(axis=-1)

    a = action[0, :, :]
    p_test, _ = env._action_to_u(a)
    p_test = p_test.cpu().numpy()
    test_num = (p_test * POP.reshape(1, -1)).sum(axis=-1)
    quara_num = quara_num[0, :, :].sum(axis=-1)
    test_ratio = test_num / POP.sum()
    quara_ratio = quara_num / POP.sum()

    days = np.arange(0, true_state.shape[0])

    fig, ax1 = plt.subplots(figsize=(28, 6))
    # fig, ax1 = plt.subplots(figsize=(16, 4.5))
    # --- 左轴：并排柱状图 ---
    # ax1.bar(days, true_state, width=0.9, alpha=0.5, color="tab:blue", label='True State')
    ax1.bar(days, true_state, width=0.9, alpha=0.5, color="tab:blue", label='真实状态')
    ax1.bar(days, partial_obs, width=0.9, alpha=0.5, color="tab:orange", label=obs_curve_label)

    ax1.set_xlabel('时间 (天)', fontproperties="SimSun")
    ax1.set_ylabel(r"现存感染规模$I_{\mathbf{combined}}$ (人)", fontproperties="SimSun")
    ax1.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6)
    ax1.set_ylim(0, 450)

    # --- 右轴：折线图（比例） ---
    ax2 = ax1.twinx()
    ax2.plot(days, test_ratio, marker='o', label=r'每日检测比例', linewidth=2)
    # ax2.plot(days, quara_ratio, marker='o', label='Quarantine Ratio', linewidth=2)
    ax2.set_ylabel('每日检测人口比例', fontproperties="SimSun")
    formatter = mtick.PercentFormatter(xmax=1.0, decimals=1)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylim(0, 0.025)

    # 在固定的几个day上画一条竖直的红色虚线
    ax1.vlines(x=[1, 3, 5, 9, 16], ymin=0, ymax=450, linestyles='--', colors='red', alpha=0.5, linewidth=1)

    # --- 合并图例 ---
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=True)
    legend.get_texts()[0].set_fontfamily('SimSun')
    legend.get_texts()[1].set_fontfamily('SimSun')
    legend.get_texts()[2].set_fontfamily('SimSun')

    plt.tight_layout()
    plt.xlim(0, true_state.shape[0])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def cal_rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def plot_community_geo_ax(ax, gdf, color_grade, show_legend=False, title=''):
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

        legend_labels.append(f"({_fmt(lo):.2f}, {_fmt(hi):.2f}]")
    # legend_labels.append(f"> {int(bins[-1]) if bins[-1].is_integer() else bins[-1]}")

    # 绘图
    plotted = gdf.plot(
        column='value',
        # cmap='RdYlGn_r',
        cmap='Greens',
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
        leg.set_bbox_to_anchor((1.5, 1), transform=ax.transAxes)
        leg.set_frame_on(False)
        leg.set_title(r"每日检测人口比例 $L_{\mathbf{test}}$")
        # 替换文本
        for text, label in zip(leg.get_texts(), legend_labels):
            text.set_text(label)
        # 方块标记
        for h in leg.legendHandles:
            h.set_marker('s')
            h.set_markersize(10)
        leg.get_title().set_fontfamily('SimSun')

    # 视觉优化：去轴、去边框、等比
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)
    ax.set_aspect('equal')

def plot_action_geo(actions, days, output_path, show_legend=False):
    day_cnt = len(days)

    # (1, day_cnt) 的地图
    fig, axs = plt.subplots(1, day_cnt, figsize=(25, 6))
    # 加载地图数据
    gdf = gpd.read_file("../../data/sz/Shenzhen_geo_data/Shenzhen_Community.shp")
    # 映射关系
    mapping = pd.read_csv("../../data/sz/community_654/mapping.csv")

    for i, day in enumerate(days):
        # 绘制子图，根据actions计算出各个区域的检测率
        a = actions[0, day, :]
        p_test, _ = env._action_to_u(a)
        p_test = p_test.cpu().numpy()
        p_test_csv = mapping.copy()
        p_test_csv['value'] = p_test[p_test_csv["filtered_idx"].values]
        print(max(p_test_csv['value']))
        new_gdf = gdf.merge(p_test_csv, left_on='OBJECTID', right_on='OBJECTID')

        # 子图标题 day，可以选择是否绘制颜色图例，自定义bins
        color_grade = [0, 1.0, 2.0, 4.0, 12]
        color_grade = [c * 0.01 for c in color_grade]
        plot_community_geo_ax(axs[i], new_gdf, color_grade, show_legend=show_legend, title=f"Day {day}")

    # 导出保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def main_draw_overall_trend():
    # # 完备上报观测也绘制一个
    # simRes0 = np.load("perfect_report/simRes.npy")
    # perfect_observation = np.load("perfect_report/obs.npy")
    # action0 = np.load("perfect_report/actions.npy")
    # quara_num0 = np.load("perfect_report/daily_quara_num.npy")
    # # plot_overall_trend(simRes0, perfect_observation, action0, quara_num0,
    # #                    "output_fig/overall_trend_perfect_obs.png",
    # #                    obs_curve_label="Perfect Reporting")
    # plot_overall_trend(simRes0, perfect_observation, action0, quara_num0,
    #                    "output_fig/overall_trend_perfect_obs.png",
    #                    obs_curve_label="稳态部分观测")

    # 部分可观测绘制一个
    simRes1 = np.load("partial_observable/simRes.npy")
    partial_observation = np.load("partial_observable/obs.npy")
    action1 = np.load("partial_observable/actions.npy")
    quara_num1 = np.load("partial_observable/daily_quara_num.npy")
    # plot_overall_trend(simRes1, partial_observation, action1, quara_num1,
    #                    "output_fig/overall_trend_partial_obs.png",
    #                    obs_curve_label="Partial Observable")
    plot_overall_trend(simRes1, partial_observation, action1, quara_num1,
                       "output_fig/overall_trend_partial_obs.png",
                       obs_curve_label="非稳态部分观测")

    # 信息重建绘制一个
    simRes2 = np.load("rebuild_info/simRes.npy")
    rebuild_state = np.load("rebuild_info/rebuild_states.npy")
    # rebuild_state第2个维度+1
    shape = list(rebuild_state.shape)
    shape[1] += 1
    rebuild_state_new = np.zeros(shape, dtype=rebuild_state.dtype)
    rebuild_state_new[:, :-1, :, :] = rebuild_state
    # rebuild_state_new[:, -1, :, :] =
    action2 = np.load("rebuild_info/actions.npy")
    quara_num2 = np.load("rebuild_info/daily_quara_num.npy")
    # plot_overall_trend(simRes2, rebuild_state_new, action2, quara_num2,
    #                    "output_fig/overall_trend_rebuild_state.png",
    #                    is_rebuild=True,
    #                    obs_curve_label="Rebuild State")
    plot_overall_trend(simRes2, rebuild_state_new, action2, quara_num2,
                       "output_fig/overall_trend_rebuild_state.png",
                       is_rebuild=True,
                       obs_curve_label="重建状态")

def main_draw_action():
    # 部分可观测
    simRes1 = np.load("partial_observable/simRes.npy")
    partial_observe = np.load("partial_observable/obs.npy")
    action1 = np.load("partial_observable/actions.npy")
    quara_num1 = np.load("partial_observable/daily_quara_num.npy")

    # 信息重建
    simRes2 = np.load("rebuild_info/simRes.npy")
    rebuild_state = np.load("rebuild_info/rebuild_states.npy")
    # rebuild_state第2个维度+1
    shape = list(rebuild_state.shape)
    shape[1] += 1
    rebuild_state_new = np.zeros(shape, dtype=rebuild_state.dtype)
    rebuild_state_new[:, :-1, :, :] = rebuild_state
    # rebuild_state_new[:, -1, :, :] =
    action2 = np.load("rebuild_info/actions.npy")
    quara_num2 = np.load("rebuild_info/daily_quara_num.npy")

    days = [1, 3, 5, 9, 16]
    plot_action_geo(action1, days, "output_fig/action_geo/partial_obs.png")
    plot_action_geo(action2, days, "output_fig/action_geo/rebuild_state.png")
    days = [3, 16]
    plot_action_geo(action1, days, "output_fig/action_geo/partial_obs_with_legend.png", show_legend=True)
    plot_action_geo(action2, days, "output_fig/action_geo/rebuild_state_with_legend.png", show_legend=True)

if __name__ == '__main__':
    main_draw_overall_trend()
    # main_draw_action()




