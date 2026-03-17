import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 16,
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)


def exponential_smoothing(series, alpha=0.3):
    """
    一阶指数平滑处理

    参数:
        series: 待平滑的序列（pd.Series）
        alpha: 平滑系数，范围(0,1)，值越小平滑效果越强
    返回:
        平滑后的序列
    """
    # 处理空序列或单元素序列
    if len(series) <= 1:
        return series.copy()

    smoothed = [series.iloc[0]]  # 初始值为序列第一个元素
    for value in series.iloc[1:]:
        # 指数平滑公式：S_t = α*y_t + (1-α)*S_{t-1}
        smoothed_val = alpha * value + (1 - alpha) * smoothed[-1]
        smoothed.append(smoothed_val)
    return pd.Series(smoothed, index=series.index)


def plot_reward_comparison(all_rewards, alpha=0.3, save_fig=True,
                           fig_path="output/消融实验-对比有无责任分发/fig.png"):
    # 1. 数据处理：先平滑
    data = all_rewards.copy()

    # 对每个组数据应用指数平滑
    data['smoothed_reward'] = data.groupby(['drd', 'step'])['reward'].transform(
        lambda x: exponential_smoothing(x, alpha=alpha)
    )

    # 2. 计算分位数
    grouped = data.groupby(['drd', 'step'])['smoothed_reward']
    quantiles = grouped.quantile([0.1, 0.5, 0.9]).unstack()

    # 3. 绘图 - 使用手动对数转换
    plt.figure(figsize=(7, 5))

    # 存储所有转换后的值，用于确定y轴范围
    all_transformed = []

    for drd_label in ['有责任分发', '无责任分发']:
        if drd_label not in quantiles.index.get_level_values('drd'):
            print(f"警告：数据中不存在 {drd_label} 的记录，跳过绘制")
            continue

        drd_data = quantiles.loc[drd_label].copy()
        steps = drd_data.index

        # 对中位数和分位曲线进行指数平滑
        # drd_data[0.5] = exponential_smoothing(drd_data[0.5])
        # drd_data[0.1] = exponential_smoothing(drd_data[0.1])
        # drd_data[0.9] = exponential_smoothing(drd_data[0.9])

        # === 关键修改：手动对数转换 ===
        # 原始值范围：(-498901.3125, -20.29735374450684)
        # 转换公式：transformed = -np.log10(-value)
        # 注意：value是负值，所以 -value 是正值
        transformed_median = -np.log10(-drd_data[0.5])
        transformed_low = -np.log10(-drd_data[0.1])
        transformed_high = -np.log10(-drd_data[0.9])

        # 收集转换后的值
        all_transformed.extend(transformed_median)
        all_transformed.extend(transformed_low)
        all_transformed.extend(transformed_high)

        color = 'blue' if drd_label == '有责任分发' else 'red'
        label = 'with response distribution' if drd_label == '有责任分发' else 'without response distribution'

        # 使用转换后的值绘图
        plt.plot(steps, transformed_median,
                 label=label,
                 linestyle='-', linewidth=2, color=color)

        # 填充区间
        plt.fill_between(steps, transformed_low, transformed_high,
                         alpha=0.1,
                         color=color)

    # 4. 坐标轴设置 - 自定义刻度
    plt.xlabel('Step')
    plt.ylabel('Reward')

    ax = plt.gca()

    # 确定合适的y轴范围（基于转换后的值）
    min_trans = np.floor(min(all_transformed))
    max_trans = np.ceil(max(all_transformed))

    # 设置y轴范围为整数刻度
    plt.ylim(min_trans, max_trans)
    # plt.ylim(-6, 0)

    # 生成主刻度位置（整数）
    ticks = np.arange(min_trans, max_trans + 1)
    ax.set_yticks(ticks)

    # 自定义刻度标签 - 显示原始负值
    def tick_formatter(transformed_val):
        """将转换后的值映射回原始负值 - 修正版本"""
        # 逆转换：original = -10**(-transformed_val)
        # 注意：这里使用 -transformed_val 而不是 transformed_val
        exponent = -transformed_val
        original_val = -10 ** exponent

        # 格式化显示
        if abs(original_val) >= 1000:
            # 使用科学计数法显示
            return f"-1e{int(exponent)}"
            # return f"-{10 ** exponent:.0e}"
        else:
            # 普通整数显示
            return f"{original_val:.0f}"

    # 应用格式化器
    ax.set_yticklabels([tick_formatter(t) for t in ticks])

    # 添加网格线
    plt.grid(alpha=0.3)

    # 确保大值在下方（原始值更负）
    # ax.invert_yaxis()

    # 5. 图例和美化
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3, which='major')
    plt.xlim(0, all_rewards['step'].max())
    plt.tight_layout()

    # 保存图片
    if save_fig:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {fig_path}")

    plt.show()

