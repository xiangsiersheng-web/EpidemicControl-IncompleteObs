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
    "font.size": 28, # Overall trend-partial-reconstruction
    # "font.size": 22, # Overall trend-perfect-partial
    # "font.size": 25, # action_geo
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Set Chinese font
# plt.rcParams['axes.unicode_minus'] = False  # Solve minus display issue

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
    # --- Left axis: side-by-side bar chart ---
    # ax1.bar(days, true_state, width=0.9, alpha=0.5, color="tab:blue", label='True State')
    ax1.bar(days, true_state, width=0.9, alpha=0.5, color="tab:blue", label='Fully observable')
    ax1.bar(days, partial_obs, width=0.9, alpha=0.5, color="tab:orange", label=obs_curve_label)

    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel(r"$I_{\mathbf{combined}}$ (persons)")
    ax1.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6)
    ax1.set_ylim(0, 450)

    # --- Right axis: line chart (ratio) ---
    ax2 = ax1.twinx()
    ax2.plot(days, test_ratio, marker='o', label=r'Daily testing level', linewidth=2)
    # ax2.plot(days, quara_ratio, marker='o', label='Quarantine Ratio', linewidth=2)
    ax2.set_ylabel('Daily testing level')
    formatter = mtick.PercentFormatter(xmax=1.0, decimals=1)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylim(0, 0.025)

    # Draw vertical red dashed lines at fixed days
    ax1.vlines(x=[1, 3, 5, 9, 16], ymin=0, ymax=450, linestyles='--', colors='red', alpha=0.5, linewidth=1)

    # --- Combine legends ---
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=True)


    plt.tight_layout()
    plt.xlim(0, true_state.shape[0])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def cal_rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))




def main_draw_overall_trend():
    # # Also plot perfect reporting observation
    # simRes0 = np.load("perfect_report/simRes.npy")
    # perfect_observation = np.load("perfect_report/obs.npy")
    # action0 = np.load("perfect_report/actions.npy")
    # quara_num0 = np.load("perfect_report/daily_quara_num.npy")
    # # plot_overall_trend(simRes0, perfect_observation, action0, quara_num0,
    # #                    "output_fig/overall_trend_perfect_obs.png",
    # #                    obs_curve_label="Perfect Reporting")
    # plot_overall_trend(simRes0, perfect_observation, action0, quara_num0,
    #                    "output_fig/overall_trend_perfect_obs.png",
    #                    obs_curve_label="Steady partial observation")

    # Plot partial observable
    simRes1 = np.load("partial_observable/simRes.npy")
    partial_observation = np.load("partial_observable/obs.npy")
    action1 = np.load("partial_observable/actions.npy")
    quara_num1 = np.load("partial_observable/daily_quara_num.npy")
    # plot_overall_trend(simRes1, partial_observation, action1, quara_num1,
    #                    "output_fig/overall_trend_partial_obs.png",
    #                    obs_curve_label="Partial Observable")
    plot_overall_trend(simRes1, partial_observation, action1, quara_num1,
                       "output_fig/overall_trend_partial_obs_en.png",
                       obs_curve_label="Non-steady partially observable")

    # Plot information reconstruction
    simRes2 = np.load("rebuild_info/simRes.npy")
    rebuild_state = np.load("rebuild_info/rebuild_states.npy")
    # rebuild_state dimension 2 + 1
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
                       "output_fig/overall_trend_rebuild_state_en.png",
                       is_rebuild=True,
                       obs_curve_label="Reconstructed state")


if __name__ == '__main__':
    main_draw_overall_trend()
    # main_draw_action()




