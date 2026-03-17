import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from matplotlib import ticker

from train_gpu import my_test, _config_args
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
import matplotlib.ticker as mtick

config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 20,
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
POP = env.POP[0]
DAY = 30

def plot_state_obs_action(state, obs, action, region_idx, output_path= None, is_perfect_report= False):
    true_state = state[0, :(DAY+1), region_idx,
                    [env.E_undetected, env.E_detected,
                     env.I_undetected, env.I_detected, env.I_reported]].sum(axis=0)
    observe = obs[0, :(DAY+1), region_idx, 0]
    a = action[0, :(DAY+1), region_idx]
    p_test, _ = env._action_to_u(a)

    # 绘图
    plt.figure(dpi=120, figsize=(7, 5))
    plt.grid(linestyle='-.', axis='both')

    plt.plot(true_state, label='真实状态', color='blue', linewidth=2, linestyle='-')
    label2 = "稳态部分观测" if is_perfect_report else "非稳态部分观测"
    plt.plot(observe, label=label2, color='orange', linewidth=2, linestyle='-')

    plt.axhline(ls='-.', color='grey')
    plt.xlim(0, DAY)
    plt.xticks(range(0, DAY + 1, 5))
    plt.ylim(0, max(true_state.max(), observe.max()) * 1.28)
    # y轴刻度最多显示一位小数
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    legend = plt.legend(fontsize=17, loc='upper left')
    for text in legend.get_texts():
        text.set_fontfamily('SimSun')
    plt.xlabel('时间 (天)', fontproperties="SimSun")
    plt.ylabel(r"现存感染规模 $I_{\mathbf{combined}}$ (人)", fontproperties="SimSun")


    plt.twinx()

    # 绘制两类动作的变化曲线
    plt.plot(p_test.cpu().numpy(), label=r"检测比例", linewidth=2, linestyle='--', color='green')
    formatter = mtick.PercentFormatter(xmax=1.0, decimals=1)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.ylim(0, max(p_test.cpu().numpy().max(), 0.02) * 1.28)

    legend = plt.legend(fontsize=17, loc='upper right')
    for text in legend.get_texts():
        text.set_fontfamily('SimSun')
    plt.ylabel(r"每日检测人口比例 $L_{\mathbf{test}}$", fontproperties="SimSun")

    # plt.title("Region " + str(region_idx))
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)

    plt.show()

    return cal_rmse(observe, true_state)

def plot_state_rebuild_action(state, rebuild, action, region_idx, output_path= None):
    true_state = state[0, :(DAY + 1), region_idx,
                 [env.E_undetected, env.E_detected,
                  env.I_undetected, env.I_detected, env.I_reported]].sum(axis=0)
    reb = rebuild[0, :(DAY + 1), region_idx, 0]
    a = action[0, :(DAY + 1), region_idx]
    p_test, _ = env._action_to_u(a)

    # 绘图
    plt.figure(dpi=120, figsize=(7, 5))
    plt.grid(linestyle='-.', axis='both')

    plt.plot(true_state, label='真实状态', color='blue', linewidth=2, linestyle='-')
    plt.plot(reb, label='重建状态', color=light_blue, linewidth=2, linestyle='-')

    plt.axhline(ls='-.', color='grey')
    plt.xlim(0, DAY)
    plt.xticks(range(0, DAY + 1, 5))
    plt.ylim(0, max(true_state.max(), reb.max()) * 1.28)
    if reb.min() < 1:
        # y轴刻度最多显示一位小数
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    legend = plt.legend(fontsize=17, loc='upper left')
    for text in legend.get_texts():
        text.set_fontfamily('SimSun')
    plt.xlabel('时间 (天)', fontproperties="SimSun")
    plt.ylabel(r"现存感染规模 $I_{\mathbf{combined}}$ (人)", fontproperties="SimSun")

    plt.twinx()

    # 绘制两类动作的变化曲线
    plt.plot(p_test.cpu().numpy(), label=r"检测比例", linewidth=2, linestyle='--',
             color='green')
    formatter = mtick.PercentFormatter(xmax=1.0, decimals=1)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.ylim(0, max(p_test.cpu().numpy()) * 1.28)

    legend = plt.legend(fontsize=17, loc='upper right')
    for text in legend.get_texts():
        text.set_fontfamily('SimSun')
    plt.ylabel(r"每日检测人口比例 $L_{\mathbf{test}}$", fontproperties="SimSun")

    # plt.title("Region " + str(region_idx))
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
    plt.show()

    return cal_rmse(reb, true_state)


def cal_rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

if __name__ == '__main__':
    state0 = np.load("perfect_report/simRes.npy")
    obs0 = np.load("perfect_report/obs.npy")
    action0 = np.load("perfect_report/actions.npy")

    state1 = np.load("partial_observable/simRes.npy")
    obs1 = np.load("partial_observable/obs.npy")
    action1 = np.load("partial_observable/actions.npy")

    state2 = np.load("rebuild_info/simRes.npy")
    rebuild2 = np.load("rebuild_info/rebuild_states.npy")
    action2 = np.load("rebuild_info/actions.npy")

    # for i in range(350, 400):
    #     if state1[0, :30, i, env.E_undetected].max() < 2 or state0[0, :30, i, env.E_undetected].max() < 1:
    #         continue
    #     rmse0 = plot_state_obs_action(state0, obs0, action0, region_idx=i)
    #     rmse1 = plot_state_obs_action(state1, obs1, action1, region_idx=i)
    #     rmse2 = plot_state_rebuild_action(state2, rebuild2, action2, region_idx=i)
    #     print(f"i={i}, rmse0={rmse0}, rmse1={rmse1}, rmse2={rmse2}")
    #     # 该区域的人口，对角线 流动紧密的几个区域
    #     column = OD[:, i]
    #     # 输出column最大的5个值以及索引
    #     print(f"i={i}, pop={POP[i]}, topk: {column.topk(5)}")
    #     print()

    for i in [359, 363]:
        if state1[0, :30, i, env.E_undetected].max() < 2 or state0[0, :30, i, env.E_undetected].max() < 1:
            continue
        if not os.path.exists(f"output_fig/region_{i}"):
            os.makedirs(f"output_fig/region_{i}")
        rmse0 = plot_state_obs_action(state0, obs0, action0, region_idx=i, output_path=f"output_fig/region_{i}/perfect_report.png", is_perfect_report= True)
        rmse1 = plot_state_obs_action(state1, obs1, action1, region_idx=i, output_path=f"output_fig/region_{i}/partial_observable.png")
        rmse2 = plot_state_rebuild_action(state2, rebuild2, action2, region_idx=i, output_path=f"output_fig/region_{i}/rebuild_info.png")
        print(f"i={i}, rmse0={rmse0}, rmse1={rmse1}, rmse2={rmse2}")
        # 该区域的人口，对角线 流动紧密的几个区域
        column = OD[:, i]
        # 输出column最大的5个值以及索引
        print(f"i={i}, pop={POP[i]}, topk: {column.topk(5)}")
        print()
