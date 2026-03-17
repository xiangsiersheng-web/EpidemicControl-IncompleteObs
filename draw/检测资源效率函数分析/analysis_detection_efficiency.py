import matplotlib.pyplot as plt
import numpy as np
import os

config = {
    "font.family": "serif",
    "font.serif": ["simsun"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    # 基础字体大小（作为基准）
    "font.size": 20,
    # 单独设置各元素的字体大小（优先级高于全局）
    "axes.labelsize": 24,      # 坐标轴标签大小
    "axes.titlesize": 20,      # 标题大小
    "legend.fontsize": 18,     # 图例大小
    "xtick.labelsize": 18,     # x轴刻度大小
    "ytick.labelsize": 18,     # y轴刻度大小
}
plt.rcParams.update(config)


def plot_detection_efficiency_function(exponent=-0.6, save=False):
    """
    绘制检测资源效率函数 ρ(x) = x^exponent

    参数:
        exponent: 指数系数，默认为-0.6
        save: 是否保存图片，默认为False
    """
    # 创建感染比例x的范围 (0, 1]，使用线性坐标
    x = np.linspace(0.001, 1, 1000)
    rho = x ** exponent

    # 绘图
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    plt.plot(x, rho, linewidth=3, color='blue')

    # 在横坐标 x=0.01 处添加标记点
    x_mark = 0.01
    rho_mark = x_mark ** exponent

    # 绘制标记点
    plt.scatter(x_mark, rho_mark, color='red', s=100, zorder=5,
                edgecolors='black', linewidth=2)

    # 添加标注
    plt.annotate(f'({x_mark:.2f}, {rho_mark:.2f})',
                 xy=(x_mark, rho_mark),
                 xytext=(x_mark + 0.05, rho_mark * 0.8),
                 fontsize=22,
                 # arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                 )

    # 使用LaTeX语法设置坐标轴标签
    plt.xlabel(r'感染比例 $x = N_{\mathbf{EI}}/N$')
    plt.ylabel(r'检测资源效率 $\rho(x)$')

    plt.xlim(0, 1)
    plt.tight_layout()

    # 保存图片
    if save:
        filename = 'detection_efficiency_function.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")

    plt.show()

    return x, rho


def plot_infection_detection_relation(exponent=-0.6, N=10000, save=False):
    """
    绘制感染人数与所需检测人数的关系（只保留第一个子图）

    参数:
        exponent: 指数系数，默认为-0.6
        N: 总人口数，默认为10000
        save: 是否保存图片，默认为False
    """
    # 创建感染比例x的范围 (0, 1]，使用线性坐标
    x = np.linspace(0.001, 1, 1000)

    # 感染人数 N_EI = x * N
    N_EI = x * N

    # 检测资源效率 ρ(x) = x^exponent
    rho = x ** exponent

    # 所需检测人数 = ρ(x) * N_EI = x^exponent * (x * N) = N * x^(exponent + 1)
    required_tests = rho * N_EI

    # 绘图（只保留一个子图）
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    plt.plot(N_EI, required_tests, linewidth=3, color='red')

    # 在横坐标 N_EI=100 处添加标记点
    N_EI_mark = 100
    # 计算对应的感染比例
    x_mark = N_EI_mark / N
    # 计算对应的所需检测人数
    required_tests_mark = N * (x_mark ** (exponent + 1))

    # 绘制标记点
    plt.scatter(N_EI_mark, required_tests_mark, color='green', s=100, zorder=5,
                edgecolors='black', linewidth=2)

    # 添加标注
    plt.annotate(f'({N_EI_mark}, {required_tests_mark:.0f})',
                 xy=(N_EI_mark, required_tests_mark),
                 xytext=(N_EI_mark + 200, required_tests_mark * 1.1),
                 fontsize=22,
                 # arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                 )

    # 使用LaTeX语法设置坐标轴标签
    plt.xlabel(r'感染人数 $N_{\mathbf{EI}}$')
    plt.ylabel(r'所需检测人数')

    plt.xlim(0, N)
    plt.tight_layout()

    # 保存图片
    if save:
        filename = 'infection_detection_relation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")

    plt.show()

    return x, N_EI, required_tests


def plot_practical_detection_rate(exponent=-0.6, L_test_values=[0.1, 0.3, 0.5, 0.8, 1.0], save=False):
    """
    绘制实际检测率 P_test 与感染比例 x 的关系（不同检测等级 L_test）

    参数:
        exponent: 指数系数，默认为-0.6
        L_test_values: 检测等级列表
        save: 是否保存图片，默认为False
    """
    # 创建感染比例x的范围 (0, 1]，使用线性坐标
    x = np.linspace(0.001, 1, 1000)

    # 绘图
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    colors = plt.cm.viridis(np.linspace(0, 1, len(L_test_values)))

    for i, L_test in enumerate(L_test_values):
        # 计算 ρ(x)
        rho = x ** exponent

        # 计算 P_test = min(L_test / (ρ(x) * x), 1)
        P_test = np.minimum(L_test / (rho * x), 1)

        plt.plot(x, P_test, linewidth=2.5, color=colors[i], label=f'$L_{{\mathrm{{test}}}} = {L_test:.1f}$')

    # 使用LaTeX语法设置坐标轴标签
    plt.xlabel(r'感染比例 $x = N_{\mathbf{EI}}/N$')
    plt.ylabel(r'实际检测率 $P_{\mathbf{test}}$')
    plt.legend(loc='upper right')
    plt.xlim(0, 1)
    plt.tight_layout()

    # 保存图片
    if save:
        filename = 'practical_detection_rate.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")

    plt.show()


def plot_detection_efficiency_function(exponent=-0.6, save=False):
    """
    绘制检测资源效率函数 ρ(x) = x^exponent

    参数:
        exponent: 指数系数，默认为-0.6
        save: 是否保存图片，默认为False
    """
    # 创建感染比例x的范围 (0, 1]，使用线性坐标
    x = np.linspace(0.001, 1, 1000)
    rho = x ** exponent

    # 绘图
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    plt.plot(x, rho, linewidth=3, color='blue')

    # 在横坐标 x=0.01 处添加标记点
    x_mark = 0.01
    rho_mark = x_mark ** exponent

    # 绘制标记点
    plt.scatter(x_mark, rho_mark, color='red', s=100, zorder=5,
                edgecolors='black', linewidth=2)

    # 添加标注
    plt.annotate(f'({x_mark:.2f}, {rho_mark:.2f})',
                 xy=(x_mark, rho_mark),
                 xytext=(x_mark + 0.05, rho_mark * 0.8),
                 fontsize=22,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 使用LaTeX语法设置坐标轴标签
    plt.xlabel(r'感染比例 $x = N_{\mathbf{EI}}/N$')
    plt.ylabel(r'检测资源效率 $\rho(x)$')

    # 添加标题
    # plt.title('(a) 检测资源效率函数', fontsize=16)

    plt.xlim(0, 1)
    plt.tight_layout()

    # 保存图片
    if save:
        filename = 'detection_efficiency_function.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")

    plt.show()

    return x, rho


def plot_infection_detection_relation(exponent=-0.6, N=10000, save=False):
    """
    绘制感染人数与所需检测人数的关系（只保留第一个子图）

    参数:
        exponent: 指数系数，默认为-0.6
        N: 总人口数，默认为10000
        save: 是否保存图片，默认为False
    """
    # 创建感染比例x的范围 (0, 1]，使用线性坐标
    x = np.linspace(0.001, 1, 1000)

    # 感染人数 N_EI = x * N
    N_EI = x * N

    # 检测资源效率 ρ(x) = x^exponent
    rho = x ** exponent

    # 所需检测人数 = ρ(x) * N_EI = x^exponent * (x * N) = N * x^(exponent + 1)
    required_tests = rho * N_EI

    # 绘图（只保留一个子图）
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    plt.plot(N_EI, required_tests, linewidth=3, color='red')

    # 在横坐标 N_EI=100 处添加标记点
    N_EI_mark = 100
    # 计算对应的感染比例
    x_mark = N_EI_mark / N
    # 计算对应的所需检测人数
    required_tests_mark = N * (x_mark ** (exponent + 1))

    # 绘制标记点
    plt.scatter(N_EI_mark, required_tests_mark, color='green', s=100, zorder=5,
                edgecolors='black', linewidth=2)

    # 添加标注
    plt.annotate(f'({N_EI_mark}, {required_tests_mark:.0f})',
                 xy=(N_EI_mark, required_tests_mark),
                 xytext=(N_EI_mark + 600, required_tests_mark * 1.2),
                 fontsize=22,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 使用LaTeX语法设置坐标轴标签
    plt.xlabel(r'感染人数 $N_{\mathbf{EI}}$ (人)')
    plt.ylabel(r'所需检测人次 (人次)')

    # 添加标题
    # plt.title('(b) 感染人数与检测需求关系', fontsize=16)

    plt.xlim(0, N)
    plt.tight_layout()

    # 保存图片
    if save:
        filename = 'infection_detection_relation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")

    plt.show()

    return x, N_EI, required_tests


def plot_actual_detected_infections(exponent=-0.6, N=10000, L_test_values=[0.1, 0.3, 0.5, 0.8, 1.0], save=False):
    """
    绘制实际检出感染人数与现存感染人数的关系（不同检测等级 L_test）

    参数:
        exponent: 指数系数，默认为-0.6
        N: 总人口数，默认为10000
        L_test_values: 检测等级列表
        save: 是否保存图片，默认为False
    """
    # 创建感染比例x的范围 (0, 1]，使用线性坐标
    x = np.linspace(0.001, 1, 1000)

    # 感染人数 N_EI = x * N
    N_EI = x * N

    # 绘图
    plt.figure(figsize=(7, 5), dpi=100)
    plt.grid(linestyle='-.', axis='both')

    colors = plt.cm.viridis(np.linspace(0, 1, len(L_test_values)))

    for i, L_test in enumerate(L_test_values):
        # 计算 ρ(x)
        rho = x ** exponent

        # 计算 P_test = min(L_test / (ρ(x) * x), 1)
        P_test = np.minimum(L_test / (rho * x), 1)

        # 计算实际检出感染人数
        # 当 P_test < 1 时：检出人数 = 总检测人次 / 每发现一个感染者所需检测人次 = (N * L_test) / ρ(x)
        # 当 P_test = 1 时：检出人数 = 总感染人数 = x * N
        actual_detected = np.where(
            P_test < 1,
            (N * L_test) / rho,  # 公式推导：N * L_test * x^0.6
            x * N  # 检测率达到100%时，所有感染者都被检出
        )

        # x轴改为现存感染人数 N_EI
        plt.plot(N_EI, actual_detected, linewidth=2.5, color=colors[i], label=f'$L_{{\mathrm{{test}}}} = {L_test:.1f}$')

    # 使用LaTeX语法设置坐标轴标签
    plt.xlabel(r'现存感染人数 $N_{\mathbf{EI}}$ (人)')
    plt.ylabel(r'实际检出感染人数 (人)')

    # 添加标题
    # plt.title('(c) 实际检出感染人数与现存感染人数关系', fontsize=16)

    plt.xlim(0, N)

    # 添加参考线：当检测能力足够时，最大可能检出人数（即所有感染者都被检出）
    # 这是对角线 y = N_EI
    plt.plot(N_EI, N_EI, '--', linewidth=1.5, color='gray', alpha=0.7,
             label='理论最大检出人数', zorder=1)

    plt.legend(loc='upper left', fontsize=17)
    plt.tight_layout()

    # 保存图片
    if save:
        filename = 'actual_detected_infections.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")

    plt.show()


# 测试代码
if __name__ == "__main__":
    # 确保当前目录存在
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")

    # 输出总名称
    print("\n" + "=" * 60)
    print("图集名称: 检测策略数学模型可视化")
    print("Figures Collection: Mathematical Model Visualization of Detection Strategies")
    print("=" * 60)

    print("\n1. (a) 检测资源效率函数")
    print("   (a) Detection Resource Efficiency Function")
    x1, rho = plot_detection_efficiency_function(exponent=-0.6, save=True)

    print("\n2. (b) 感染人数与检测需求关系")
    print("   (b) Relationship between Infection Number and Detection Demand")
    x2, N_EI, required_tests = plot_infection_detection_relation(exponent=-0.6, N=10000, save=True)

    print("\n3. (c) 实际检出感染人数与感染比例关系")
    print("   (c) Relationship between Actual Detected Infections and Infection Proportion")
    plot_actual_detected_infections(exponent=-0.6, N=10000, L_test_values=[0.1, 0.3, 0.5, 0.7], save=True)

    # 输出一些关键数据点
    print("\n关键数据点:")
    print(f"{'感染比例':<10} {'感染人数':<10} {'所需检测人数':<15}")
    print("-" * 50)
    for percentage in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        idx = np.argmin(np.abs(x2 - percentage))
        print(f"{percentage * 100:>6.1f}%  {N_EI[idx]:>9.0f}  {required_tests[idx]:>14.0f}")