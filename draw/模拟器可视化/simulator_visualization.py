"""将模拟器可视化"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import math

import glob
import re
import imageio.v3 as iio

from train_gpu import my_test, _config_args
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1

config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 15,
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)

args = _config_args()
args.env_data_dir = '../../data/'
env = EpidemicModel(args, env_count=1)
OD = env.OD[0]
POP = env.POP[0]

_superscript = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

def _fmt(x):
    def fmt_sci(x, sig=3, omit_one=True, superscript=True):
        """
        x: 数值
        sig: 有效数字个数
        omit_one: 当系数a≈1时是否省略“1×”
        superscript: 是否将指数显示为上标
        """
        if x == 0:
            return "0"

        # 计算指数 n 与系数 a，使得 x = a * 10^n，且 a ∈ [1, 10)
        n = math.floor(math.log10(abs(x)))
        a = x / (10 ** n)

        # 按有效数字四舍五入
        # 例如 sig=3 时，a 保留 3 位有效数字
        fmt = f"{{:.{sig - 1}g}}"
        a_str = fmt.format(a)

        # 处理可能出现的 "10"（例如 9.999≈10）的边界
        if a_str in ("10", "10.0"):
            a_str = "1"
            n += 1

        # 构造指数
        if superscript:
            sup_map = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
            n_str = str(n).translate(sup_map)
            exp_str = f"10{n_str}"
        else:
            exp_str = f"10^{n}"

        # 是否省略“1×”
        if omit_one and a_str in ("1", "1.0"):
            return exp_str
        else:
            # 使用乘号 ×（更易读）
            return f"{a_str}×{exp_str}"

    return fmt_sci(float(x), sig=1, omit_one=False, superscript=True)


def plot_community_geo_ax(ax, gdf, color_grade, show_legend=False, legend_title=""):
    bins = [float(b) for b in color_grade]
    bins_eps = [b + 1e-10 for b in bins]

    # 生成图例标签
    # 采用左开右闭区间的描述，更贴合 mapclassify.UserDefined 的行为
    legend_labels = []
    legend_labels.append(f"≤ {_fmt(bins[0])}")
    for i in range(1, len(bins)):
        lo = bins[i - 1]
        hi = bins[i]

        legend_labels.append(f"({_fmt(lo)}, {_fmt(hi)}]")
    legend_labels.append(f"> {_fmt(bins[-1])}")

    # 绘图
    plotted = gdf.plot(
        column='value',
        cmap='RdYlGn_r',
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
        leg.set_bbox_to_anchor((1.2, 0.9), transform=ax.transAxes)
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

def plot_group_figures(simRes, output_path):
    # 加载地图数据
    gdf = gpd.read_file("../../data/sz/Shenzhen_geo_data/Shenzhen_Community.shp")
    # 映射关系
    mapping = pd.read_csv("../../data/sz/community_654/mapping.csv")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    DAY = simRes.shape[1]
    infections = simRes[0, :, :, :][:, :, [env.E_undetected, env.E_detected,
                                           env.I_undetected, env.I_detected, env.I_reported]].sum(axis=-1)
    MAX = np.max( infections)
    # color_grade 指数增长分割为6个值，最小0，最大80000
    color_grade = [0, 1e2, 5e2, 2e3, 1e4, 8e4]
    #
    for day in range(DAY):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        state = infections[day]
        state_csv = mapping.copy()
        state_csv['value'] = state[state_csv["filtered_idx"].values]
        new_gdf = gdf.merge(state_csv, left_on='OBJECTID', right_on='OBJECTID')
        plot_community_geo_ax(ax, new_gdf, color_grade, legend_title="", show_legend= True)
        ax.set_title("Day " + str(day))
        plt.tight_layout()
        plt.savefig(output_path+'/day_'+str(day)+'.png', dpi=200)
        plt.show()


def output_animation(
    input_dir="group_figures",
    mp4_path="animation.mp4",
    gif_path=None,
    fps=12,
    pattern="day_*.png",
    gif_every_n=1,  # GIF 可以抽帧减小体积，例如取 2 表示每隔 2 帧取一帧
):
    """
    将 input_dir 中由 plot_group_figures 导出的帧图，按帧率合成为 MP4（可选同时导出 GIF）。

    参数
    ----
    input_dir : str
        帧图片所在目录。
    mp4_path : str
        导出的 MP4 文件路径。
    gif_path : str 或 None
        若不为 None，则同时导出 GIF 到该路径。
    fps : int 或 float
        帧率（每秒帧数）。
    pattern : str
        匹配帧图片的通配符，默认匹配 'day_*.png'。
    gif_every_n : int
        导出 GIF 时的抽帧步长，1 表示不抽帧，2 表示每隔一帧取一帧。
    """

    # 1) 收集并自然排序（按 day_数字.png 的数字排序）
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        raise FileNotFoundError(f"未在 {input_dir} 中找到匹配 {pattern} 的图片。")

    def _day_key(p):
        m = re.search(r"day_(\d+)\.png$", os.path.basename(p))
        return int(m.group(1)) if m else float("inf")

    files = sorted(files, key=_day_key)

    # 2) 读入所有帧（保证尺寸一致；plot_group_figures 保存的图片尺寸已一致）
    frames = [iio.imread(f) for f in files]

    # 3) 写出 MP4（体积小、清晰）
    iio.imwrite(
        mp4_path,
        frames,
        fps=fps,
        codec="libx264",        # 常用编码器
        quality=7,              # 画质 0-10（数值越大质量越高）
    )

    # 4) 可选写出 GIF（浏览器易展示，但体积较大）
    if gif_path is not None:
        gif_frames = frames[::max(1, int(gif_every_n))]
        iio.imwrite(
            gif_path,
            gif_frames,
            duration=1.0 / fps,  # GIF 用每帧时长控制播放速度
            loop=0               # 循环播放
        )

    print(f"已生成：{mp4_path}" + (f" 和 {gif_path}" if gif_path else ""))


if __name__ == '__main__':
    # simRes = np.load("simRes.npy")
    # plot_group_figures(simRes, "group_figures")

    output_animation(
        input_dir="group_figures",
        mp4_path="animation.mp4",
        gif_path="animation.gif",  # 如不需要 GIF 可设为 None
        fps=12,
        gif_every_n=1  # GIF 抽帧减小体积，可按需调整
    )