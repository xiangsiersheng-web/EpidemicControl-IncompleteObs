import math
import numpy as np
import matplotlib.pyplot as plt

# 配置字体
config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
}
plt.rcParams.update(config)



print(math.log(1/3) / math.log(0.1))

def f(x):
    return 1 / (x ** (0.7))


for xi in [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
    print(xi, f(xi), xi * f(xi))


# 绘制一个曲线图，横轴为x（感染占比），纵轴为检测单个感染者所需的检测人数
# 图上标注出一些值，比如：x=1e-5, 1e-4, 1e-3, 1e-2, 1e-1

# 生成不同的感染占比值（0.00001到1之间）
x_values = np.linspace(1e-5, 1.0, 100)

# 计算 f(x) 和 x*f(x) 值
f_values = f(x_values)
x_f_values = x_values * f_values

# 设置关键点
key_points = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# 绘制第一张图：x 与 f(x) 的关系
plt.figure(figsize=(8, 6))
plt.plot(x_values, f_values, label=r'$f(x) = \frac{1}{x^{0.5}}$', color='blue')
# 标记关键点
for point in key_points:
    plt.plot(point, f(point), 'ro')  # 红点标记
    plt.text(point, f(point), f'({point:.0e}, {f(point):.2f})', fontsize=12, verticalalignment='bottom')
plt.xlabel('Infection Ratio (x)', fontsize=14)
plt.ylabel('Tests per Infected Person f(x)', fontsize=14)
plt.title('Relationship between Infection Ratio and f(x)', fontsize=16)
plt.grid(True)
plt.legend()

# 显示第一张图
plt.show()

# 绘制第二张图：x 与 x*f(x) 的关系
plt.figure(figsize=(8, 6))
plt.plot(x_values, x_f_values, label=r'$x \cdot f(x)$', color='red')
# 标记关键点
for point in key_points:
    plt.plot(point, point * f(point), 'go')  # 绿色点标记
    plt.text(point, point * f(point), f'({point:.0e}, {point * f(point):.2f})', fontsize=12, verticalalignment='bottom')
plt.xlabel('Infection Ratio (x)', fontsize=14)
plt.ylabel('Number of Tests per Infected Person x * f(x)', fontsize=14)
plt.title('Relationship between Infection Ratio and x * f(x)', fontsize=16)
plt.grid(True)
plt.legend()

# 显示第二张图
plt.show()