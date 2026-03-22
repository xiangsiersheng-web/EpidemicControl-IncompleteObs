import numpy as np
import matplotlib.pyplot as plt
import os

from environment.uncertain_seir_vector_v4 import EpidemicModel
from train_gpu import _config_args

# Set plot style
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
    Plot natural transmission curves for infected persons under high and low transmission scenarios

    Args:
        high_data_path: High transmission scenario data path
        low_data_path: Low transmission scenario data path
        title: Chart title
        save_path: Save path
        figsize: Chart size
    """
    # Define colors
    high_red = '#FF6B6B'  # Red represents high transmission
    low_red = '#FFB6C1'  # Light red represents low transmission
    fill_alpha = 0.15  # Fill transparency

    # Load data
    high_simRes = np.load(high_data_path)  # Shape: (num_seeds, time_steps, zone_num, state_dim)
    low_simRes = np.load(low_data_path)  # Shape: (num_seeds, time_steps, zone_num, state_dim)

    # Define infected status indices (adjust according to actual model definition)
    # Here we assume infected includes: E_undetected, E_detected, I_undetected, I_detected, I_reported
    # Please adjust according to your model status indices
    infected_indices = [env.I_detected, env.I_reported, env.I_undetected]  # Please adjust as needed

    # Calculate total infected count for each seed at each time step
    high_infected = high_simRes[:, :, :, infected_indices].sum(axis=(2, 3))
    low_infected = low_simRes[:, :, :, infected_indices].sum(axis=(2, 3))

    # Calculate statistics
    def calculate_stats(data_2d):
        mean = np.nanmean(data_2d, axis=0)
        p10 = np.nanpercentile(data_2d, 10, axis=0)
        p90 = np.nanpercentile(data_2d, 90, axis=0)
        return mean, p10, p90

    high_mean, high_p10, high_p90 = calculate_stats(high_infected)
    low_mean, low_p10, low_p90 = calculate_stats(low_infected)

    # Time axis
    t_len = high_mean.shape[0]
    x = np.arange(t_len)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot high transmission scenario (red)
    ax.fill_between(x, high_p10, high_p90, color=high_red, alpha=fill_alpha)
    ax.plot(x, high_mean, linewidth=2.5, color=high_red, label=r'$R_0=5.5$')

    # Plot low transmission scenario (light red)
    ax.fill_between(x, low_p10, low_p90, color=low_red, alpha=fill_alpha)
    ax.plot(x, low_mean, linewidth=2.5, color=low_red, label=r'$R_0=2.6$')

    # Set chart properties
    # ax.set_xlabel("Day")
    # # ax.set_ylabel(r"$I_{\mathbf{combined}}$")
    # ax.set_ylabel(r"Current Infections")
    ax.set_xlabel("Time (days)")
    # ax.set_ylabel(r"$I_{\mathbf{combined}}$")
    ax.set_ylabel(r"Current infections $I$ count (persons)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set legend
    ax.legend(frameon=True, framealpha=0.9, loc='upper right')

    # Set axis
    ax.margins(x=0.01)
    ax.set_xlim(0, t_len - 1)
    ax.set_ylim(0, None)

    # Scientific notation for y-axis
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Add grid and ticks
    ax.minorticks_on()

    # Add peak information
    peak_high = np.max(high_mean)
    peak_low = np.max(low_mean)
    peak_day_high = np.argmax(high_mean)
    peak_day_low = np.argmax(low_mean)


    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {save_path}")
    print(f"High transmission peak: {peak_high:.2e} (day {peak_day_high})")
    print(f"Low transmission peak: {peak_low:.2e} (day {peak_day_low})")

    return fig, ax


# Usage example
if __name__ == "__main__":
    plot_transmission_curves()