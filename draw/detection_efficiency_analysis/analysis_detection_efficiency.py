import matplotlib.pyplot as plt
import numpy as np
import os

config = {
    "font.family": "serif",
    "font.serif": ["simsun"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    # Base font size (as reference)
    "font.size": 20,
    # Set font size for individual elements (higher priority than global)
    "axes.labelsize": 24,      # Axis label size
    "axes.titlesize": 20,      # Title size
    "legend.fontsize": 18,     # Legend size
    "xtick.labelsize": 18,     # x-axis tick size
    "ytick.labelsize": 18,     # y-axis tick size
}
plt.rcParams.update(config)


def plot_detection_efficiency_function(exponent=-0.6, save=False):
    """
    Plot detection resource efficiency function ρ(x) = x^exponent

    Args:
        exponent: Exponent coefficient, default is -0.6
        save: Whether to save the figure, default is False
    """
    # Create range of infection proportion x (0, 1], using linear coordinates
    x = np.linspace(0.001, 1, 1000)
    rho = x ** exponent

    # Plot
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    plt.plot(x, rho, linewidth=3, color='blue')

    # Add marker at x=0.01
    x_mark = 0.01
    rho_mark = x_mark ** exponent

    # Plot marker point
    plt.scatter(x_mark, rho_mark, color='red', s=100, zorder=5,
                edgecolors='black', linewidth=2)

    # Add annotation
    plt.annotate(f'({x_mark:.2f}, {rho_mark:.2f})',
                 xy=(x_mark, rho_mark),
                 xytext=(x_mark + 0.05, rho_mark * 0.8),
                 fontsize=22,
                 # arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                 )

    # Use LaTeX syntax for axis labels
    plt.xlabel(r'Infection proportion $x = N_{\mathbf{EI}}/N$')
    plt.ylabel(r'Detection resource efficiency $\rho(x)$')

    plt.xlim(0, 1)
    plt.tight_layout()

    # Save figure
    if save:
        filename = 'detection_efficiency_function.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {filename}")

    plt.show()

    return x, rho


def plot_infection_detection_relation(exponent=-0.6, N=10000, save=False):
    """
    Plot relationship between infection count and required detection count (only first subplot)

    Args:
        exponent: Exponent coefficient, default is -0.6
        N: Total population, default is 10000
        save: Whether to save the figure, default is False
    """
    # Create range of infection proportion x (0, 1], using linear coordinates
    x = np.linspace(0.001, 1, 1000)

    # Number of infected N_EI = x * N
    N_EI = x * N

    # Detection resource efficiency ρ(x) = x^exponent
    rho = x ** exponent

    # Required detection count = ρ(x) * N_EI = x^exponent * (x * N) = N * x^(exponent + 1)
    required_tests = rho * N_EI

    # Plot (only one subplot)
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    plt.plot(N_EI, required_tests, linewidth=3, color='red')

    # Add marker at N_EI=100
    N_EI_mark = 100
    # Calculate corresponding infection proportion
    x_mark = N_EI_mark / N
    # Calculate corresponding required detection count
    required_tests_mark = N * (x_mark ** (exponent + 1))

    # Plot marker point
    plt.scatter(N_EI_mark, required_tests_mark, color='green', s=100, zorder=5,
                edgecolors='black', linewidth=2)

    # Add annotation
    plt.annotate(f'({N_EI_mark}, {required_tests_mark:.0f})',
                 xy=(N_EI_mark, required_tests_mark),
                 xytext=(N_EI_mark + 200, required_tests_mark * 1.1),
                 fontsize=22,
                 # arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                 )

    # Use LaTeX syntax for axis labels
    plt.xlabel(r'Number of infected $N_{\mathbf{EI}}$')
    plt.ylabel(r'Required detection count')

    plt.xlim(0, N)
    plt.tight_layout()

    # Save figure
    if save:
        filename = 'infection_detection_relation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {filename}")

    plt.show()

    return x, N_EI, required_tests


def plot_practical_detection_rate(exponent=-0.6, L_test_values=[0.1, 0.3, 0.5, 0.8, 1.0], save=False):
    """
    Plot relationship between actual detection rate P_test and infection proportion x (different detection levels L_test)

    Args:
        exponent: Exponent coefficient, default is -0.6
        L_test_values: List of detection levels
        save: Whether to save the figure, default is False
    """
    # Create range of infection proportion x (0, 1], using linear coordinates
    x = np.linspace(0.001, 1, 1000)

    # Plot
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    colors = plt.cm.viridis(np.linspace(0, 1, len(L_test_values)))

    for i, L_test in enumerate(L_test_values):
        # Calculate ρ(x)
        rho = x ** exponent

        # Calculate P_test = min(L_test / (ρ(x) * x), 1)
        P_test = np.minimum(L_test / (rho * x), 1)

        plt.plot(x, P_test, linewidth=2.5, color=colors[i], label=f'$L_{{\mathrm{{test}}}} = {L_test:.1f}$')

    # Use LaTeX syntax for axis labels
    plt.xlabel(r'Infection proportion $x = N_{\mathbf{EI}}/N$')
    plt.ylabel(r'Actual detection rate $P_{\mathbf{test}}$')
    plt.legend(loc='upper right')
    plt.xlim(0, 1)
    plt.tight_layout()

    # Save figure
    if save:
        filename = 'practical_detection_rate.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {filename}")

    plt.show()


def plot_detection_efficiency_function(exponent=-0.6, save=False):
    """
    Plot detection resource efficiency function ρ(x) = x^exponent

    Args:
        exponent: Exponent coefficient, default is -0.6
        save: Whether to save the figure, default is False
    """
    # Create range of infection proportion x (0, 1], using linear coordinates
    x = np.linspace(0.001, 1, 1000)
    rho = x ** exponent

    # Plot
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    plt.plot(x, rho, linewidth=3, color='blue')

    # Add marker at x=0.01
    x_mark = 0.01
    rho_mark = x_mark ** exponent

    # Plot marker point
    plt.scatter(x_mark, rho_mark, color='red', s=100, zorder=5,
                edgecolors='black', linewidth=2)

    # Add annotation
    plt.annotate(f'({x_mark:.2f}, {rho_mark:.2f})',
                 xy=(x_mark, rho_mark),
                 xytext=(x_mark + 0.05, rho_mark * 0.8),
                 fontsize=22,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Use LaTeX syntax for axis labels
    plt.xlabel(r'Infection proportion $x = N_{\mathbf{EI}}/N$')
    plt.ylabel(r'Detection resource efficiency $\rho(x)$')

    # Add title
    # plt.title('(a) Detection resource efficiency function', fontsize=16)

    plt.xlim(0, 1)
    plt.tight_layout()

    # Save figure
    if save:
        filename = 'detection_efficiency_function.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {filename}")

    plt.show()

    return x, rho


def plot_infection_detection_relation(exponent=-0.6, N=10000, save=False):
    """
    Plot relationship between infection count and required detection count (only first subplot)

    Args:
        exponent: Exponent coefficient, default is -0.6
        N: Total population, default is 10000
        save: Whether to save the figure, default is False
    """
    # Create range of infection proportion x (0, 1], using linear coordinates
    x = np.linspace(0.001, 1, 1000)

    # Number of infected N_EI = x * N
    N_EI = x * N

    # Detection resource efficiency ρ(x) = x^exponent
    rho = x ** exponent

    # Required detection count = ρ(x) * N_EI = x^exponent * (x * N) = N * x^(exponent + 1)
    required_tests = rho * N_EI

    # Plot (only one subplot)
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    plt.plot(N_EI, required_tests, linewidth=3, color='red')

    # Add marker at N_EI=100
    N_EI_mark = 100
    # Calculate corresponding infection proportion
    x_mark = N_EI_mark / N
    # Calculate corresponding required detection count
    required_tests_mark = N * (x_mark ** (exponent + 1))

    # Plot marker point
    plt.scatter(N_EI_mark, required_tests_mark, color='green', s=100, zorder=5,
                edgecolors='black', linewidth=2)

    # Add annotation
    plt.annotate(f'({N_EI_mark}, {required_tests_mark:.0f})',
                 xy=(N_EI_mark, required_tests_mark),
                 xytext=(N_EI_mark + 600, required_tests_mark * 1.2),
                 fontsize=22,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Use LaTeX syntax for axis labels
    plt.xlabel(r'Number of infected $N_{\mathbf{EI}}$ (persons)')
    plt.ylabel(r'Required detection count (person-times)')

    # Add title
    # plt.title('(b) Relationship between infection count and detection demand', fontsize=16)

    plt.xlim(0, N)
    plt.tight_layout()

    # Save figure
    if save:
        filename = 'infection_detection_relation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {filename}")

    plt.show()

    return x, N_EI, required_tests


def plot_actual_detected_infections(exponent=-0.6, N=10000, L_test_values=[0.1, 0.3, 0.5, 0.8, 1.0], save=False):
    """
    Plot relationship between actual detected infections and current infections (different detection levels L_test)

    Args:
        exponent: Exponent coefficient, default is -0.6
        N: Total population, default is 10000
        L_test_values: List of detection levels
        save: Whether to save the figure, default is False
    """
    # Create range of infection proportion x (0, 1], using linear coordinates
    x = np.linspace(0.001, 1, 1000)

    # Number of infected N_EI = x * N
    N_EI = x * N

    # Plot
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    colors = plt.cm.viridis(np.linspace(0, 1, len(L_test_values)))

    for i, L_test in enumerate(L_test_values):
        # Calculate ρ(x)
        rho = x ** exponent

        # Calculate P_test = min(L_test / (ρ(x) * x), 1)
        P_test = np.minimum(L_test / (rho * x), 1)

        # Calculate actual detected infections
        # When P_test < 1: detected = total tests / tests per infection = (N * L_test) / ρ(x)
        # When P_test = 1: detected = total infections = x * N
        actual_detected = np.where(
            P_test < 1,
            (N * L_test) / rho,  # Formula derivation: N * L_test * x^0.6
            x * N  # When detection rate reaches 100%, all infections are detected
        )

        # x-axis is current infections N_EI
        plt.plot(N_EI, actual_detected, linewidth=2.5, color=colors[i], label=f'$L_{{\mathrm{{test}}}} = {L_test:.1f}$')

    # Use LaTeX syntax for axis labels
    plt.xlabel(r'Current infections $N_{\mathbf{EI}}$ (persons)')
    plt.ylabel(r'Actual detected infections (persons)')

    # Add title
    # plt.title('(c) Relationship between actual detected infections and current infections', fontsize=16)

    plt.xlim(0, N)

    # Add reference line: maximum possible detections when testing capacity is sufficient (all infections detected)
    # This is the diagonal y = N_EI
    plt.plot(N_EI, N_EI, '--', linewidth=1.5, color='gray', alpha=0.7,
             label='Theoretical max detections', zorder=1)

    plt.legend(loc='upper left', fontsize=17)
    plt.tight_layout()

    # Save figure
    if save:
        filename = 'actual_detected_infections.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {filename}")

    plt.show()


# Test code
if __name__ == "__main__":
    # Ensure current directory exists
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Output general name
    print("\n" + "=" * 60)
    print("Figure Collection: Mathematical Model Visualization of Detection Strategies")
    print("=" * 60)

    print("\n1. (a) Detection Resource Efficiency Function")
    x1, rho = plot_detection_efficiency_function(exponent=-0.6, save=True)

    print("\n2. (b) Relationship between Infection Number and Detection Demand")
    x2, N_EI, required_tests = plot_infection_detection_relation(exponent=-0.6, N=10000, save=True)

    print("\n3. (c) Relationship between Actual Detected Infections and Infection Proportion")
    plot_actual_detected_infections(exponent=-0.6, N=10000, L_test_values=[0.1, 0.3, 0.5, 0.7], save=True)

    # Output some key data points
    print("\nKey data points:")
    print(f"{'Infection proportion':<20} {'Infection count':<15} {'Required tests':<15}")
    print("-" * 60)
    for percentage in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        idx = np.argmin(np.abs(x2 - percentage))
        print(f"{percentage * 100:>10.1f}%  {N_EI[idx]:>14.0f}  {required_tests[idx]:>14.0f}")