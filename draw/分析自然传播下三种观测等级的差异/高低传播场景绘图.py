import numpy as np
import matplotlib.pyplot as plt
import os

from environment.uncertain_seir_vector_v4 import EpidemicModel
from train_gpu import _config_args

# 设置绘图风格
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

def plot_transmission_curves(
        high_data_path="data/simRes_high.npy",
        low_data_path="data/simRes_low.npy",
        title="",
        save_path="figure/different_R0_transmission_comparison.png",
        figsize=(7, 5)
):
    """
    绘制高低传播场景下的感染者自然传播曲线

    参数:
        high_data_path: 高传播场景数据路径
        low_data_path: 低传播场景数据路径
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    # 定义颜色
    high_red = '#FF6B6B'  # 红色代表高传播
    low_red = '#FFB6C1'  # 浅红色代表低传播
    fill_alpha = 0.15  # 填充透明度

    # 加载数据
    high_simRes = np.load(high_data_path)  # 形状: (num_seeds, time_steps, zone_num, state_dim)
    low_simRes = np.load(low_data_path)  # 形状: (num_seeds, time_steps, zone_num, state_dim)

    # 定义感染者状态索引（根据实际模型定义调整）
    # 这里假设感染者包括: E_undetected, E_detected, I_undetected, I_detected, I_reported
    # 请根据您的模型状态索引进行调整
    infected_indices = [env.I_detected, env.I_reported, env.I_undetected]  # 请根据实际调整

    # 计算每个种子每个时间步的总感染者数
    high_infected = high_simRes[:, :, :, infected_indices].sum(axis=(2, 3))
    low_infected = low_simRes[:, :, :, infected_indices].sum(axis=(2, 3))

    # 计算统计量
    def calculate_stats(data_2d):
        mean = np.nanmean(data_2d, axis=0)
        p10 = np.nanpercentile(data_2d, 10, axis=0)
        p90 = np.nanpercentile(data_2d, 90, axis=0)
        return mean, p10, p90

    high_mean, high_p10, high_p90 = calculate_stats(high_infected)
    low_mean, low_p10, low_p90 = calculate_stats(low_infected)

    # 时间轴
    t_len = high_mean.shape[0]
    x = np.arange(t_len)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制高传播场景（红色）
    ax.fill_between(x, high_p10, high_p90, color=high_red, alpha=fill_alpha)
    ax.plot(x, high_mean, linewidth=2.5, color=high_red, label=r'$R_0=5.5$')

    # 绘制低传播场景（浅红色）
    ax.fill_between(x, low_p10, low_p90, color=low_red, alpha=fill_alpha)
    ax.plot(x, low_mean, linewidth=2.5, color=low_red, label=r'$R_0=2.6$')

    # 设置图表属性
    # ax.set_xlabel("Day")
    # # ax.set_ylabel(r"$I_{\mathbf{combined}}$")
    # ax.set_ylabel(r"Current Infections")
    ax.set_xlabel("时间 (天)", fontproperties="SimSun")
    # ax.set_ylabel(r"$I_{\mathbf{combined}}$")
    ax.set_ylabel(r"当前感染者$I$ 数量 (人)", fontproperties="SimSun")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 设置图例
    ax.legend(frameon=True, framealpha=0.9, loc='upper right')

    # 设置坐标轴
    ax.margins(x=0.01)
    ax.set_xlim(0, t_len - 1)
    ax.set_ylim(0, None)

    # 科学计数法显示y轴
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # 添加网格和刻度
    ax.minorticks_on()

    # 添加峰值信息
    peak_high = np.max(high_mean)
    peak_low = np.max(low_mean)
    peak_day_high = np.argmax(high_mean)
    peak_day_low = np.argmax(low_mean)


    plt.tight_layout()

    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"图表已保存至: {save_path}")
    print(f"高传播峰值: {peak_high:.2e} (第{peak_day_high}天)")
    print(f"低传播峰值: {peak_low:.2e} (第{peak_day_low}天)")

    return fig, ax


# 使用示例
if __name__ == "__main__":
    plot_transmission_curves()