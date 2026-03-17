import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd

config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 16,
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)

def _plot_missing_rate_predict_rmse_scatter(missing_rate_predict_rmse_table_file_path="",
                                             y_log=False,
                                             log_base=10,
                                             clip_min=1e-6,
                                             y_max=200,
                                             y_min=10,  # 若为 None，则自动计算一个正的下界
                                             output_path="./缺失率_RMSE.png"
                                             ):
    from matplotlib.ticker import LogLocator, LogFormatter

    df = pd.read_excel(missing_rate_predict_rmse_table_file_path).copy()

    # 对数轴需要正值，先做抬升
    if y_log:
        df["predict_total_effect"] = np.where(df["predict_total_effect"] <= 0, clip_min, df["predict_total_effect"])

    plt.figure(figsize=(8, 6))

    markers = ['o', 's', 'v', 'D', 'H', '^']

    # 采样比例
    sample_ratio = 0.3  # 只画30%的点

    for i, name in enumerate(label_map.keys()):
        sub = df[df["name"] == name]
        marker = markers[i % len(markers)]
        if sub.empty:
            continue

        # 方法1：按missing_rate分箱，然后在每个箱中均匀采样
        # 创建10个等宽的missing_rate区间
        sub_copy = sub.copy()
        sub_copy['rate_bin'] = pd.cut(sub_copy['missing_rate'], bins=20)

        # 从每个箱中采样相同数量的点
        sampled_points = []
        for _, group in sub_copy.groupby('rate_bin'):
            if len(group) > 0:
                # 计算每个箱需要采样的点数
                n_samples = max(1, int(len(group) * sample_ratio))
                sampled = group.sample(min(n_samples, len(group)), random_state=42)
                sampled_points.append(sampled)

        if sampled_points:
            sub_sampled = pd.concat(sampled_points)
        else:
            sub_sampled = sub.sample(frac=sample_ratio, random_state=42)

        plt.scatter(
            sub_sampled["missing_rate"], sub_sampled["predict_total_effect"],
            s=60, alpha=0.98, edgecolors="none", label=label_map[name],
            marker=marker,
        )

    # plt.xlabel("Missing Rate")
    plt.xlabel(r"$MR_{\mathbf{obs}}$", fontproperties="SimSun")
    plt.ylabel(r"$\mathbf{RMSE}_{\mathbf{global}}$", fontproperties="SimSun")
    plt.xlim(0, 1)

    ax = plt.gca()

    if y_log:
        ax.set_yscale('log', base=log_base)

        # 自动确定下界（保证为正，且不高于上界）
        if y_min is None:
            # 取本次绘图中所有点的最小正值，适度留一点空隙
            positive_scores = df["predict_total_effect"][df["predict_total_effect"] > 0]
            y_min_auto = positive_scores.min() if not positive_scores.empty else clip_min
            y_min_auto = max(clip_min, y_min_auto * 0.9)
            y_min_final = min(y_min_auto, y_max)  # 防止极端情况下下界超过上界
        else:
            y_min_final = max(clip_min, float(y_min))

        ax.set_ylim(y_min_final, float(y_max))

        # 对数主刻度与格式
        ax.yaxis.set_major_locator(LogLocator(base=log_base))
        ax.yaxis.set_major_formatter(LogFormatter(base=log_base))
    else:
        # 线性坐标时也按需限制
        ax.set_ylim(float(y_min), float(y_max))

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()




def _plot_missing_rate_score_scatter(missing_rate_score_table_file_path="",
                                     y_log=True,
                                     log_base=10,
                                     clip_min=1e-6,
                                     y_max=10000,
                                     y_min=1,
                                     output_path="./缺失率_Score.png"):
    from matplotlib.ticker import LogLocator, LogFormatter, ScalarFormatter

    df = pd.read_excel(missing_rate_score_table_file_path).copy()

    # 对数轴需要正值，先做抬升
    if y_log:
        df["score"] = np.where(df["score"] <= 0, clip_min, df["score"])

    fig, ax = plt.subplots(figsize=(8, 6))

    markers = ['o', 's', 'v', 'D', 'H', '^']

    # 采样比例
    sample_ratio = 0.3  # 只画30%的点

    for i, name in enumerate(label_map.keys()):
        sub = df[df["name"] == name]
        marker = markers[i % len(markers)]
        if sub.empty:
            continue

        # 方法1：按missing_rate分箱，然后在每个箱中均匀采样
        # 创建10个等宽的missing_rate区间
        sub_copy = sub.copy()
        sub_copy['rate_bin'] = pd.cut(sub_copy['missing_rate'], bins=20)

        # 从每个箱中采样相同数量的点
        sampled_points = []
        for _, group in sub_copy.groupby('rate_bin'):
            if len(group) > 0:
                # 计算每个箱需要采样的点数
                n_samples = max(1, int(len(group) * sample_ratio))
                sampled = group.sample(min(n_samples, len(group)), random_state=42)
                sampled_points.append(sampled)

        if sampled_points:
            sub_sampled = pd.concat(sampled_points)
        else:
            sub_sampled = sub.sample(frac=sample_ratio, random_state=42)

        plt.scatter(
            sub_sampled["missing_rate"], sub_sampled["score"],
            s=60, alpha=0.98, edgecolors="none", label=label_map[name],
            marker=marker,
        )

    ax.set_xlabel(r"$MR_{\mathbf{obs}}$", fontproperties="SimSun")
    ax.set_ylabel(r"$\mathbf{Score}$", fontproperties="SimSun")
    ax.set_xlim(0, 1)

    if y_log:
        ax.set_yscale('log', base=log_base)

        # 自动确定下界
        if y_min is None:
            positive_scores = df["score"][df["score"] > 0]
            y_min_auto = positive_scores.min() if not positive_scores.empty else clip_min
            y_min_auto = max(clip_min, y_min_auto * 0.9)
            y_min_final = min(y_min_auto, y_max)
        else:
            y_min_final = max(clip_min, float(y_min))

        ax.set_ylim(y_min_final, float(y_max))

        # 对数坐标的格式化
        ax.yaxis.set_major_locator(LogLocator(base=log_base))
        ax.yaxis.set_major_formatter(LogFormatter(base=log_base))
    else:
        # 线性坐标
        ax.set_ylim(float(y_min), float(y_max))

        # 关键步骤：设置科学计数法
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 4))

        # 可选：自定义科学计数法格式
        ax.yaxis.get_offset_text().set_fontsize(10)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


# label的对应关系
label_map = {
    "Certain": "Steady Partially Observable",
    "Uncertain": "No Reconstruction(NPO)",
    "ode_formula": "PureODE(NPO)",
    "idw": "IDW(NPO)",
    "gnn_gru_ordinary": "GCN-GRU(NPO)",
    "gnn_gru_agent": "ODE-DynNet(NPO)",
}


from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 20,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
}
plt.rcParams.update(config)


def plot_missing_rate_scatter_with_quantiles():
    """缺失率受到观测缺失比例-持续时间影响，绘制10-90分位图"""
    missing_rate_file = "missing_rate_table_high.xlsx"

    # 读取数据
    df = pd.read_excel(missing_rate_file)

    # 过滤出name为Certain的数据
    df_certain = df[df["name"] == "Certain"].copy()

    # 计算平均缺失比例和平均缺失持续时间
    df_certain["avg_mask_rate"] = (df_certain["mask_rate_up"] + df_certain["mask_rate_down"]) / 2
    df_certain["avg_mask_duration"] = (df_certain["mask_duration_up"] + df_certain["mask_duration_down"]) / 2

    # 按照平均缺失持续时间分组
    duration_groups = df_certain.groupby("avg_mask_duration")

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 获取颜色映射 - 使用viridis表示持续时间
    cmap = get_cmap("viridis")

    # 获取所有唯一的持续时间值并排序
    durations = sorted(df_certain["avg_mask_duration"].unique())

    # 为每个持续时间分配颜色
    norm_durations = [(d - min(durations)) / (max(durations) - min(durations))
                      if max(durations) > min(durations) else 0.5 for d in durations]
    colors = [cmap(norm) for norm in norm_durations]

    # 绘制每个持续时间的折线和分位带
    lines = []
    labels = []

    for i, (duration, group) in enumerate(duration_groups):
        # 按照平均缺失比例分组，计算每个组的统计量
        grouped_stats = group.groupby("avg_mask_rate")["missing_rate"].agg([
            ('mean', 'mean'),
            ('p10', lambda x: np.percentile(x, 10)),
            ('p90', lambda x: np.percentile(x, 90)),
            ('count', 'count')
        ]).reset_index()

        # 按照平均缺失比例排序
        grouped_stats = grouped_stats.sort_values("avg_mask_rate")

        # 提取数据
        x = grouped_stats["avg_mask_rate"]
        mean = grouped_stats["mean"]
        p10 = grouped_stats["p10"]
        p90 = grouped_stats["p90"]
        count = grouped_stats["count"]

        # 绘制10-90分位带
        ax.fill_between(
            x, p10, p90,
            color=colors[i],
            alpha=0.15,  # 较低的透明度，避免遮盖其他线条
            edgecolor='none',
            label=f'持续时间={duration:.0f} (10-90分位)'
        )

        # 绘制均值折线
        line, = ax.plot(
            x, mean,
            marker='o',
            markersize=6,
            linewidth=2.5,
            color=colors[i],
            label=f"持续时间={duration:.0f} (均值)"
        )

        # 在某些点上标记数据数量（可选）
        # 选择几个点标记数据数量
        if len(x) > 0:
            # 标记第一个点
            ax.annotate(f'n={count.iloc[0]}',
                        xy=(x.iloc[0], mean.iloc[0]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=colors[i])

        lines.append(line)
        # labels.append(f"持续时间={duration:.0f}")
        labels.append(r"$MD={duration:.0f}$")

    # 设置图表属性
    ax.set_xlabel(r"平均缺失比例 $r$", fontname='SimSun')
    ax.set_ylabel(r"观测缺失率 $MR_{\mathbf{obs}}$", fontname='SimSun')

    # 添加网格
    ax.grid(True, linestyle="--", alpha=0.4)

    # 设置图例 - 简化版本，避免太多图例项
    # 只显示均值线的图例
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, duration in enumerate(durations):
        legend_elements.append(Line2D([0], [0],
                                      color=colors[i],
                                      lw=2.5,
                                      markersize=8,
                                      label=fr"$t={duration:.0f}$"))

    legend = ax.legend(handles=legend_elements,
              frameon=True,
              framealpha=0.9,
              loc="best",
              fontsize=20,
              title="平均缺失持续时间 (天)"
              )
    legend.get_title().set_fontname('SimSun')
    for text in legend.get_texts():
        text.set_fontname('SimSun')

    # 设置坐标轴范围
    ax.set_xlim(-0.001, 1)  # 稍微扩展一下x轴范围
    ax.set_ylim(-0.001, 1)  # 留出15%的空间

    plt.tight_layout()

    # 保存图表
    output_path = "./缺失率_比例_持续时间_分位图.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"图表已保存至: {output_path}")

    # 输出详细的统计信息
    print("\n详细统计信息:")
    print(f"数据总数: {len(df_certain)}")
    print(f"不同的持续时间数量: {len(durations)}")
    print(f"持续时间范围: {min(durations):.0f} - {max(durations):.0f}")

    # 为每个持续时间输出统计信息
    print("\n各持续时间组统计:")
    for i, (duration, group) in enumerate(duration_groups):
        print(f"\n持续时间 {duration:.0f}:")
        print(f"  数据点数量: {len(group)}")
        print(f"  平均缺失率: {group['missing_rate'].mean():.6f}")
        print(f"  缺失率范围: {group['missing_rate'].min():.6f} - {group['missing_rate'].max():.6f}")
        print(f"  标准差: {group['missing_rate'].std():.6f}")

def plot_missing_rate_scatter():
    """缺失率受到观测缺失比例-持续时间影响"""
    missing_rate_file = "missing_rate_table_high.xlsx"

    # 读取数据
    df = pd.read_excel(missing_rate_file)

    # 过滤出name为Certain的数据
    df_certain = df[df["name"] == "Certain"].copy()

    # 计算平均缺失比例和平均缺失持续时间
    df_certain["avg_mask_rate"] = (df_certain["mask_rate_up"] + df_certain["mask_rate_down"]) / 2
    df_certain["avg_mask_duration"] = (df_certain["mask_duration_up"] + df_certain["mask_duration_down"]) / 2

    # 按照平均缺失持续时间分组
    duration_groups = df_certain.groupby("avg_mask_duration")

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 获取颜色映射 - 使用viridis表示持续时间
    cmap = get_cmap("viridis")

    # 获取所有唯一的持续时间值并排序
    durations = sorted(df_certain["avg_mask_duration"].unique())

    # 为每个持续时间分配颜色
    norm_durations = [(d - min(durations)) / (max(durations) - min(durations))
                      if max(durations) > min(durations) else 0.5 for d in durations]
    colors = [cmap(norm) for norm in norm_durations]

    # 绘制每个持续时间的折线
    lines = []
    labels = []

    for i, (duration, group) in enumerate(duration_groups):
        # 按照平均缺失比例排序并计算每个缺失比例下的平均缺失率
        group_sorted = group.sort_values("avg_mask_rate")

        # 对相同的平均缺失比例取均值（处理重复实验）
        agg_data = group_sorted.groupby("avg_mask_rate")["missing_rate"].mean().reset_index()

        # 绘制折线
        line, = ax.plot(
            agg_data["avg_mask_rate"],
            agg_data["missing_rate"],
            marker='o',
            markersize=8,
            linewidth=2.5,
            color=colors[i],
            label=f"Duration={duration:.0f}"
        )

        lines.append(line)
        labels.append(f"Duration={duration:.0f}")

    # 设置图表属性
    ax.set_xlabel("Average Missing Proportion", fontsize=14)
    ax.set_ylabel("Missing Rate", fontsize=14)
    ax.set_title("Missing Rate vs Average Missing Proportion\nfor Different Missing Durations", fontsize=16, pad=15)

    # 添加网格
    ax.grid(True, linestyle="--", alpha=0.6)

    # 设置图例
    ax.legend(frameon=True, framealpha=0.9, loc="best", fontsize=12)

    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, df_certain["missing_rate"].max() * 1.1)  # 留出10%的空间

    # 添加颜色条表示持续时间
    # 由于我们已经使用了图例，颜色条不是必须的，但可以作为备选方案
    # 如果需要颜色条，可以取消注释下面的代码
    """
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(durations), vmax=max(durations)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Missing Duration', fontsize=12)
    """

    plt.tight_layout()

    # 保存图表
    output_path = "./缺失率_比例_持续时间.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"图表已保存至: {output_path}")

    # 输出一些统计信息
    print("\n统计信息:")
    print(f"数据总数: {len(df_certain)}")
    print(f"不同的持续时间数量: {len(durations)}")
    print(f"不同的缺失比例数量: {len(df_certain['avg_mask_rate'].unique())}")
    print(f"缺失率范围: {df_certain['missing_rate'].min():.6f} - {df_certain['missing_rate'].max():.6f}")


if __name__ == "__main__":
    _plot_missing_rate_score_scatter("./missing_rate_table_high.xlsx",
                                     output_path="./缺失率_Score_high.png")

    _plot_missing_rate_predict_rmse_scatter("./missing_rate_table_high.xlsx",
                                     output_path="./缺失率_RMSE_high.png")

    _plot_missing_rate_score_scatter("./missing_rate_table_low.xlsx",
                                     output_path="./缺失率_Score_low.png")

    _plot_missing_rate_predict_rmse_scatter("./missing_rate_table_low.xlsx",
                                            output_path="./缺失率_RMSE_low.png")

    # plot_missing_rate_scatter()
    plot_missing_rate_scatter_with_quantiles()