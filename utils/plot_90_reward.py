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
    First-order exponential smoothing

    Args:
        series: Sequence to be smoothed (pd.Series)
        alpha: Smoothing coefficient, range (0,1), smaller values give stronger smoothing
    Returns:
        Smoothed sequence
    """
    # Handle empty or single-element sequences
    if len(series) <= 1:
        return series.copy()

    smoothed = [series.iloc[0]]  # Initial value is the first element of the sequence
    for value in series.iloc[1:]:
        # Exponential smoothing formula: S_t = α*y_t + (1-α)*S_{t-1}
        smoothed_val = alpha * value + (1 - alpha) * smoothed[-1]
        smoothed.append(smoothed_val)
    return pd.Series(smoothed, index=series.index)


def plot_reward_comparison(all_rewards, alpha=0.3, save_fig=True,
                           fig_path="output/ablation_responsibility_distribution/fig.png"):
    # 1. Data processing: smooth first
    data = all_rewards.copy()

    # Apply exponential smoothing to each group of data
    data['smoothed_reward'] = data.groupby(['drd', 'step'])['reward'].transform(
        lambda x: exponential_smoothing(x, alpha=alpha)
    )

    # 2. Calculate quantiles
    grouped = data.groupby(['drd', 'step'])['smoothed_reward']
    quantiles = grouped.quantile([0.1, 0.5, 0.9]).unstack()

    # 3. Plot - using manual logarithmic transformation
    plt.figure(figsize=(7, 5))

    # Store all transformed values for determining y-axis range
    all_transformed = []

    for drd_label in ['with responsibility distribution', 'without responsibility distribution']:
        if drd_label not in quantiles.index.get_level_values('drd'):
            print(f"Warning: No records found for {drd_label}, skipping plot")
            continue

        drd_data = quantiles.loc[drd_label].copy()
        steps = drd_data.index

        # Apply exponential smoothing to median and quantile curves
        # drd_data[0.5] = exponential_smoothing(drd_data[0.5])
        # drd_data[0.1] = exponential_smoothing(drd_data[0.1])
        # drd_data[0.9] = exponential_smoothing(drd_data[0.9])

        # === Key modification: manual logarithmic transformation ===
        # Original value range: (-498901.3125, -20.29735374450684)
        # Transformation formula: transformed = -np.log10(-value)
        # Note: value is negative, so -value is positive
        transformed_median = -np.log10(-drd_data[0.5])
        transformed_low = -np.log10(-drd_data[0.1])
        transformed_high = -np.log10(-drd_data[0.9])

        # Collect transformed values
        all_transformed.extend(transformed_median)
        all_transformed.extend(transformed_low)
        all_transformed.extend(transformed_high)

        color = 'blue' if drd_label == 'with responsibility distribution' else 'red'
        label = 'with response distribution' if drd_label == 'with responsibility distribution' else 'without response distribution'

        # Plot using transformed values
        plt.plot(steps, transformed_median,
                 label=label,
                 linestyle='-', linewidth=2, color=color)

        # Fill the interval
        plt.fill_between(steps, transformed_low, transformed_high,
                         alpha=0.1,
                         color=color)

    # 4. Axis settings - custom ticks
    plt.xlabel('Step')
    plt.ylabel('Reward')

    ax = plt.gca()

    # Determine appropriate y-axis range (based on transformed values)
    min_trans = np.floor(min(all_transformed))
    max_trans = np.ceil(max(all_transformed))

    # Set y-axis range to integer ticks
    plt.ylim(min_trans, max_trans)
    # plt.ylim(-6, 0)

    # Generate major tick positions (integers)
    ticks = np.arange(min_trans, max_trans + 1)
    ax.set_yticks(ticks)

    # Custom tick labels - display original negative values
    def tick_formatter(transformed_val):
        """Map transformed values back to original negative values - corrected version"""
        # Inverse transformation: original = -10**(-transformed_val)
        # Note: use -transformed_val instead of transformed_val
        exponent = -transformed_val
        original_val = -10 ** exponent

        # Format display
        if abs(original_val) >= 1000:
            # Display using scientific notation
            return f"-1e{int(exponent)}"
            # return f"-{10 ** exponent:.0e}"
        else:
            # Regular integer display
            return f"{original_val:.0f}"

    # Apply formatter
    ax.set_yticklabels([tick_formatter(t) for t in ticks])

    # Add grid lines
    plt.grid(alpha=0.3)

    # Ensure larger values are at the bottom (original values more negative)
    # ax.invert_yaxis()

    # 5. Legend and beautification
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3, which='major')
    plt.xlim(0, all_rewards['step'].max())
    plt.tight_layout()

    # Save figure
    if save_fig:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")

    plt.show()

