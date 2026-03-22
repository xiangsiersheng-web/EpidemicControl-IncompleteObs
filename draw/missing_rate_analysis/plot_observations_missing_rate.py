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
                                             y_min=10,  # If None, automatically calculate a positive lower bound
                                             output_path="./missing_rate_RMSE.png"
                                             ):
    from matplotlib.ticker import LogLocator, LogFormatter

    df = pd.read_excel(missing_rate_predict_rmse_table_file_path).copy()

    # Logarithmic axis needs positive values, first apply lifting
    if y_log:
        df["predict_total_effect"] = np.where(df["predict_total_effect"] <= 0, clip_min, df["predict_total_effect"])

    plt.figure(figsize=(8, 6))

    for name in label_map.keys():
        sub = df[df["name"] == name]
        if sub.empty:
            continue
        plt.scatter(
            sub["missing_rate"], sub["predict_total_effect"],
            s=24, alpha=0.75, edgecolors="none", label=label_map[name]
        )

    # plt.xlabel("Missing Rate")
    plt.xlabel(r"$MR_{\mathbf{obs}}$")
    plt.ylabel(r"$\mathbf{RMSE}_{\mathbf{global}}$")
    plt.xlim(0, 1)

    ax = plt.gca()

    if y_log:
        ax.set_yscale('log', base=log_base)

        # Automatically determine lower bound (ensure positive, not higher than upper bound)
        if y_min is None:
            # Take minimum positive value from all points in this plot, leave some gap
            positive_scores = df["predict_total_effect"][df["predict_total_effect"] > 0]
            y_min_auto = positive_scores.min() if not positive_scores.empty else clip_min
            y_min_auto = max(clip_min, y_min_auto * 0.9)
            y_min_final = min(y_min_auto, y_max)  # Prevent lower bound from exceeding upper bound in extreme cases
        else:
            y_min_final = max(clip_min, float(y_min))

        ax.set_ylim(y_min_final, float(y_max))

        # Logarithmic major ticks and format
        ax.yaxis.set_major_locator(LogLocator(base=log_base))
        ax.yaxis.set_major_formatter(LogFormatter(base=log_base))
    else:
        # Linear coordinates also need constraints
        ax.set_ylim(float(y_min), float(y_max))

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def _plot_missing_rate_score_scatter(missing_rate_score_table_file_path="",
                                     y_log=True,
                                     log_base=10,
                                     clip_min=1e-6,
                                     y_max=10000,
                                     y_min=1,  # If None, automatically calculate a positive lower bound
                                     output_path="./missing_rate_Score.png"
                                     ):
    from matplotlib.ticker import LogLocator, LogFormatter

    df = pd.read_excel(missing_rate_score_table_file_path).copy()

    # Logarithmic axis needs positive values, first apply lifting
    if y_log:
        df["score"] = np.where(df["score"] <= 0, clip_min, df["score"])

    plt.figure(figsize=(8, 6))

    for name in label_map.keys():
        sub = df[df["name"] == name]
        if sub.empty:
            continue
        plt.scatter(
            sub["missing_rate"], sub["score"],
            s=24, alpha=0.75, edgecolors="none", label=label_map[name]
        )

    # plt.xlabel("Missing Rate")
    plt.xlabel(r"$MR_{\mathbf{obs}}$")
    plt.ylabel("Score")
    plt.xlim(0, 1)

    ax = plt.gca()

    if y_log:
        ax.set_yscale('log', base=log_base)

        # Automatically determine lower bound (ensure positive, not higher than upper bound)
        if y_min is None:
            # Take minimum positive value from all points in this plot, leave some gap
            positive_scores = df["score"][df["score"] > 0]
            y_min_auto = positive_scores.min() if not positive_scores.empty else clip_min
            y_min_auto = max(clip_min, y_min_auto * 0.9)
            y_min_final = min(y_min_auto, y_max)  # Prevent lower bound from exceeding upper bound in extreme cases
        else:
            y_min_final = max(clip_min, float(y_min))

        ax.set_ylim(y_min_final, float(y_max))

        # Logarithmic major ticks and format
        ax.yaxis.set_major_locator(LogLocator(base=log_base))
        ax.yaxis.set_major_formatter(LogFormatter(base=log_base))
    else:
        # Linear coordinates also need constraints
        ax.set_ylim(float(y_min), float(y_max))

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# label mapping relationship
label_map = {
    "Certain": "Perfect Reporting",
    "Uncertain": "No Reconstruction(PO)",
    "ode_formula": "PureODE(PO)",
    "idw": "IDW(PO)",
    "gnn_gru_ordinary": "GCN-GRU(PO)",
    "gnn_gru_agent": "ODE-DynNet(PO)",
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
    """Missing rate is affected by observation missing proportion-duration, plot 10-90 quantile chart"""
    missing_rate_file = "missing_rate_table_high.xlsx"

    # Read data
    df = pd.read_excel(missing_rate_file)

    # Filter data where name is Certain
    df_certain = df[df["name"] == "Certain"].copy()

    # Calculate average missing proportion and average missing duration
    df_certain["avg_mask_rate"] = (df_certain["mask_rate_up"] + df_certain["mask_rate_down"]) / 2
    df_certain["avg_mask_duration"] = (df_certain["mask_duration_up"] + df_certain["mask_duration_down"]) / 2

    # Group by average missing duration
    duration_groups = df_certain.groupby("avg_mask_duration")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get color mapping - use viridis for duration
    cmap = get_cmap("viridis")

    # Get all unique duration values and sort
    durations = sorted(df_certain["avg_mask_duration"].unique())

    # Assign color for each duration
    norm_durations = [(d - min(durations)) / (max(durations) - min(durations))
                      if max(durations) > min(durations) else 0.5 for d in durations]
    colors = [cmap(norm) for norm in norm_durations]

    # Plot lines and quantile bands for each duration
    lines = []
    labels = []

    for i, (duration, group) in enumerate(duration_groups):
        # Group by average missing proportion, calculate statistics for each group
        grouped_stats = group.groupby("avg_mask_rate")["missing_rate"].agg([
            ('mean', 'mean'),
            ('p10', lambda x: np.percentile(x, 10)),
            ('p90', lambda x: np.percentile(x, 90)),
            ('count', 'count')
        ]).reset_index()

        # Sort by average missing proportion
        grouped_stats = grouped_stats.sort_values("avg_mask_rate")

        # Extract data
        x = grouped_stats["avg_mask_rate"]
        mean = grouped_stats["mean"]
        p10 = grouped_stats["p10"]
        p90 = grouped_stats["p90"]
        count = grouped_stats["count"]

        # Plot 10-90 quantile band
        ax.fill_between(
            x, p10, p90,
            color=colors[i],
            alpha=0.15,  # Lower transparency to avoid obscuring other lines
            edgecolor='none',
            label=f'Duration={duration:.0f} (10-90%)'
        )

        # Plot mean line
        line, = ax.plot(
            x, mean,
            marker='o',
            markersize=6,
            linewidth=2.5,
            color=colors[i],
            label=f"Duration={duration:.0f} (Mean)"
        )

        # Mark data count on some points (optional)
        # Select a few points to mark data count
        if len(x) > 0:
            # Mark the first point
            ax.annotate(f'n={count.iloc[0]}',
                        xy=(x.iloc[0], mean.iloc[0]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=colors[i])

        lines.append(line)
        labels.append(f"Duration={duration:.0f}")

    # Set chart properties
    ax.set_xlabel("Average Missing Proportion")
    # ax.set_ylabel("Missing Rate")
    ax.set_ylabel(r"$MR_{\mathbf{obs}}$")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.4)

    # Set legend - simplified version to avoid too many legend items
    # Only show mean line legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, duration in enumerate(durations):
        legend_elements.append(Line2D([0], [0],
                                      color=colors[i],
                                      lw=2.5,
                                      # marker='o',
                                      markersize=8,
                                      label=f'Duration={duration:.0f}'),
                                       # label=f'{duration:.0f}'),
                )

    ax.legend(handles=legend_elements,
              frameon=True,
              framealpha=0.9,
              loc="best",
              fontsize=20,
              title="Average Missing Duration")

    # Set axis range
    # ax.set_xlim(-0.02, 1.02)  # Slightly expand x-axis range
    ax.set_xlim(-0.001, 1)  # Slightly Expand x-axis range

    # Ensure y-axis starts from 0, but consider the upper bound of quantile band
    y_max = df_certain["missing_rate"].max() * 1.15
    # ax.set_ylim(-0.001, y_max)  # Leave 15% space
    ax.set_ylim(-0.001, 1)  # Leave 15% space

    # # Add explanatory text
    # ax.text(0.02, 0.98,
    #         "Shaded areas represent 10-90 percentile ranges\nPoints show mean values",
    #         transform=ax.transAxes,
    #         fontsize=11,
    #         verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_path = "./missing_rate_proportion_duration_quantile_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {output_path}")

    # Output detailed statistics
    print("\nDetailed statistics:")
    print(f"Total data: {len(df_certain)}")
    print(f"Number of different durations: {len(durations)}")
    print(f"Duration range: {min(durations):.0f} - {max(durations):.0f}")

    # Output statistics for each duration
    print("\nStatistics by duration group:")
    for i, (duration, group) in enumerate(duration_groups):
        print(f"\nDuration {duration:.0f}:")
        print(f"  Number of data points: {len(group)}")
        print(f"  Average missing rate: {group['missing_rate'].mean():.6f}")
        print(f"  Missing rate range: {group['missing_rate'].min():.6f} - {group['missing_rate'].max():.6f}")
        print(f"  Standard deviation: {group['missing_rate'].std():.6f}")

def plot_missing_rate_scatter():
    """Missing rate is affected by observation missing proportion-duration"""
    missing_rate_file = "missing_rate_table_high.xlsx"

    # Read data
    df = pd.read_excel(missing_rate_file)

    # Filter data where name is Certain
    df_certain = df[df["name"] == "Certain"].copy()

    # Calculate average missing proportion and average missing duration
    df_certain["avg_mask_rate"] = (df_certain["mask_rate_up"] + df_certain["mask_rate_down"]) / 2
    df_certain["avg_mask_duration"] = (df_certain["mask_duration_up"] + df_certain["mask_duration_down"]) / 2

    # Group by average missing duration
    duration_groups = df_certain.groupby("avg_mask_duration")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get color mapping - use viridis for duration
    cmap = get_cmap("viridis")

    # Get all unique duration values and sort
    durations = sorted(df_certain["avg_mask_duration"].unique())

    # Assign color for each duration
    norm_durations = [(d - min(durations)) / (max(durations) - min(durations))
                      if max(durations) > min(durations) else 0.5 for d in durations]
    colors = [cmap(norm) for norm in norm_durations]

    # Plot lines for each duration
    lines = []
    labels = []

    for i, (duration, group) in enumerate(duration_groups):
        # Sort by average missing proportion and calculate mean missing rate for each proportion
        group_sorted = group.sort_values("avg_mask_rate")

        # Take mean for same average missing proportion (handle repeated experiments)
        agg_data = group_sorted.groupby("avg_mask_rate")["missing_rate"].mean().reset_index()

        # Plot line
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

    # Set chart properties
    ax.set_xlabel("Average Missing Proportion", fontsize=14)
    ax.set_ylabel("Missing Rate", fontsize=14)
    ax.set_title("Missing Rate vs Average Missing Proportion\nfor Different Missing Durations", fontsize=16, pad=15)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.6)

    # Set legend
    ax.legend(frameon=True, framealpha=0.9, loc="best", fontsize=12)

    # Set axis range
    ax.set_xlim(0, 1)
    ax.set_ylim(0, df_certain["missing_rate"].max() * 1.1)  # Leave 10% space

    # Add color bar for duration
    # Since we already use legend, color bar is not required but can be an alternative
    # Uncomment the following code if color bar is needed
    """
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(durations), vmax=max(durations)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Missing Duration', fontsize=12)
    """

    plt.tight_layout()

    # Save figure
    output_path = "./missing_rate_proportion_duration.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {output_path}")

    # Output some statistics
    print("\nStatistics:")
    print(f"Total data: {len(df_certain)}")
    print(f"Number of different durations: {len(durations)}")
    print(f"Number of different missing proportions: {len(df_certain['avg_mask_rate'].unique())}")
    print(f"Missing rate range: {df_certain['missing_rate'].min():.6f} - {df_certain['missing_rate'].max():.6f}")


if __name__ == "__main__":
    # _plot_missing_rate_score_scatter("./missing_rate_table_high.xlsx",
    #                                  output_path="./missing_rate_Score_high.png")
    #
    # _plot_missing_rate_predict_rmse_scatter("./missing_rate_table_high.xlsx",
    #                                  output_path="./missing_rate_RMSE_high.png")
    #
    # _plot_missing_rate_score_scatter("./missing_rate_table_low.xlsx",
    #                                  output_path="./missing_rate_Score_low.png")
    #
    # _plot_missing_rate_predict_rmse_scatter("./missing_rate_table_low.xlsx",
    #                                         output_path="./missing_rate_RMSE_low.png")

    # plot_missing_rate_scatter()
    plot_missing_rate_scatter_with_quantiles()