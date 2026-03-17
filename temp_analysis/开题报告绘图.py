from config import args
from environment.meta_env_tensor_vector import EpidemicModelTensorVector
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.ticker import ScalarFormatter


config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 16,
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)


args.env_data_dir = "../data/"
args.device_name = "cpu"

def draw_all_state():
    # 定义环境
    env = EpidemicModelTensorVector(args, env_count=1)
    env.reset()

    # 模拟环境
    while True:
        env.step()
        if env.day > env.period:
            break
    simRes = env.simRes[0]

    # 绘图
    all_state = simRes.sum(dim = 1) # (period + 1, 4)
    t = np.linspace(0, env.period, env.period + 1)
    S, E, I, R = all_state[:, 0], all_state[:, 1], all_state[:, 2], all_state[:, 3]

    plt.figure(figsize=(6, 5))
    plt.plot(t, S, label='S (Susceptible)', color='blue', lw=2)
    plt.plot(t, E, label='E (Exposed)', color='orange', lw=2)
    plt.plot(t, I, label='I (Infected)', color='red', lw=2)
    plt.plot(t, R, label='R (Recovered)', color='green', lw=2)
    plt.xlim(0, env.period)
    plt.ylim(0, )
    plt.xlabel('Time (Days)')
    plt.ylabel('Population')
    plt.title('ODE Model Dynamics')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()

    # 绘制地图
    shp_file = "../data/sz/Shenzhen_geo_data/Shenzhen_Street.shp"
    shp_data = gpd.read_file(shp_file)
    day = I.argmax() - 1
    shp_data['Infections'] = simRes[day, :, 2]

    # shp_data.plot(column='Infections', cmap='Reds', legend=True)
    # plt.title(f'Infections on Day {day}')
    # plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    shp_data.plot(column='Infections', ax=ax, cmap='Reds', legend=True,
                  legend_kwds={'label': "Number of Infections"})
    ax.set_title(f'Infections on Day {day}')
    plt.show()


def draw_obs_imperfect():
    # 1.观测不完全
    args.I_obs_imperfect = True
    args.I_obs_imperfect_down = 0.3
    args.I_obs_imperfect_up = 0.6
    env = EpidemicModelTensorVector(args, env_count=1)
    env.reset()

    # 模拟环境
    while True:
        env.step()
        if env.day > env.period:
            break
    I = env.simRes[0, :, :, 2] # (period + 1, zone_num)
    I_imperfect = env.imperfect_obs[0, :, :, 2] # (period + 1, zone_num)

    region_idx = 0
    t = range(env.period + 1)  # 创建时间轴

    # 绘制观测完全的I和不完全的I
    # 创建图形
    plt.figure(figsize=(6, 5))
    plt.plot(t, I[:, region_idx], label='True Infections', color='blue', lw=2)
    plt.plot(t, I_imperfect[:, region_idx], label='Imperfect Observations', color='red', lw=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Infections')
    # 标题为区域I的感染人数
    plt.title(f'Infections in Region {region_idx}')
    plt.grid(True)
    plt.xlim(0, env.period)
    plt.ylim(0, )
    plt.legend()

    # 设置 y 轴为科学计数法
    ax = plt.gca()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # 设置为科学计数法的阈值
    ax.yaxis.set_major_formatter(formatter)

    plt.show()


def draw_abnormal_event():
    # 异常聚集事件
    env = EpidemicModelTensorVector(args, env_count=1)

    env.reset()
    # 模拟环境
    while True:
        env.step()
        if env.day > env.period:
            break
    I_normal = env.simRes[0, :, :, 2]

    env.reset()
    while True:
        old_betas = env.betas.clone()
        if env.day in range(25, 27):
            new_betas = old_betas.clone()
            # new_betas[:, 0] = 2 * old_betas[:, 0]
            env.adjust_contagious_OD(regions = [0], increase_ratio = [10])
        else:
            new_betas = old_betas.clone()
        env.betas = new_betas

        env.step()
        env.betas = old_betas
        env.reset_contagious_OD()
        if env.day > env.period:
            break
    I_abnormal = env.simRes[0, :, :, 2]

    region_idx = 0
    t = range(env.period + 1)  # 创建时间轴

    plt.figure(figsize=(6, 5))
    plt.plot(t, I_normal[:, region_idx], label='Normal Transmission', color='blue', lw=2)
    plt.plot(t, I_abnormal[:, region_idx], label='Abnormal Transmission', color='red', lw=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Infections')
    plt.title(f'Infections in Region {region_idx}')
    plt.grid(True)
    plt.xlim(0, env.period)
    plt.ylim(0, )
    plt.legend()

    # 设置 y 轴为科学计数法
    ax = plt.gca()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # 设置为科学计数法的阈值
    ax.yaxis.set_major_formatter(formatter)

    plt.show()



def draw_param_change():
    # 绘制参数变化的影响
    env = EpidemicModelTensorVector(args, env_count=1)

    env.reset()
    # 模拟环境
    while True:
        env.step()
        if env.day > env.period:
            break
    I_fixed = env.simRes[0, :, :, 2]

    env.reset()
    betas = env.betas.clone()
    while True:
        # if env.day == 10:
        #     env.betas = 1.5 * env.betas

        # if env.day == 25:
        #     env.betas = 0.6 * betas
        env.betas = betas + torch.randn(env.betas.shape) * 0.1

        env.step()
        env.reset_contagious_OD()
        if env.day > env.period:
            break
    I_change = env.simRes[0, :, :, 2]

    region_idx = 0
    t = range(env.period + 1)  # 创建时间轴

    plt.figure(figsize=(6, 5))
    plt.plot(t, I_fixed[:, region_idx], label='Fixed Beta', color='blue', lw=2)
    plt.plot(t, I_change[:, region_idx], label='Changed Beta', color='red', lw=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Infections')
    plt.title(f'Infections in Region {region_idx}')
    plt.grid(True)
    plt.xlim(0, env.period)
    plt.ylim(0, )
    # plt.tight_layout()
    plt.legend()

    # 设置 y 轴为科学计数法
    ax = plt.gca()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # 设置为科学计数法的阈值
    ax.yaxis.set_major_formatter(formatter)

    plt.show()



def draw_uncertain_action():
    # action的不确定
    env = EpidemicModelTensorVector(args, env_count=1)
    actions = torch.zeros(env.env_count, env.period + 1, env.ZONE_NUM)
    actions[:, 20: 50, :] = 2

    env.reset()
    # 模拟环境
    while True:
        if env.day in range(21, 51):
            action = actions[:, env.day - 1, :]
            env.step(action=action)
        else:
            env.step()
        if env.day > env.period:
            break
    I_expected = env.simRes[0, :, :, 2]
    a_expected = actions.clone()

    env.reset()
    # 模拟环境
    while True:
        if env.day in range(21, 51):
            action = actions[:, env.day - 1, :].clone()
            action = action - 0.01 * ((env.day - 20) ** 1.2) + 0.01 * torch.randn(action.shape)
            env.step(action=action)
            actions[:, env.day - 2, :] = action
        else:
            env.step()
        if env.day > env.period:
            break
    I_true = env.simRes[0, :, :, 2]
    a_true = actions.clone()

    region_idx = 0
    t = range(env.period + 1)  # 创建时间轴

    plt.figure(figsize=(6, 5))
    plt.plot(t, I_expected[:, region_idx], label='Expected Infections', color='blue', lw=2)
    plt.plot(t, I_true[:, region_idx], label='True Infections', color='red', lw=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Infections')
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(0, )
    plt.grid(True)
    # 设置 y 轴为科学计数法
    ax = plt.gca()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # 设置为科学计数法的阈值
    ax.yaxis.set_major_formatter(formatter)

    plt.twinx()
    plt.plot(t, a_expected[0, :, region_idx], label='Expected Action', color='blue', lw=1, alpha=0.8, linestyle='--')
    plt.plot(t, a_true[0, :, region_idx], label='True Action', color='red', lw=1, alpha=0.8, linestyle='--')
    plt.ylabel('Action')

    plt.yticks(range(0, 3, 1))
    plt.ylim(-1, 3)


    plt.title(f'Infections in Region {region_idx}')
    plt.xlim(0, env.period)
    plt.legend(fontsize=12, loc='upper right')



    plt.show()


def draw_uncertain_transition():
    # 读取csv文件 sz_2024-09-19_20-32_4_high.csv
    df_certain = pd.read_csv('sz_2024-09-19_20-32_4_high.csv')
    # dynamic_env_sz_2024-09-03_20-53_4_high_betachange=spatial.csv
    df_uncertain = pd.read_csv('dynamic_env_sz_2024-09-03_20-53_4_high_betachange=spatial.csv')

    # 按照step，value绘制奖励变化
    # 设置画布
    plt.figure(figsize=(7, 5))

    # 绘制确定性环境下的奖励变化
    plt.plot(df_certain['Step'], df_certain['Value'], label='State Certain Transition', marker='o')

    # 绘制不确定性环境下的奖励变化
    plt.plot(df_uncertain['Step'], df_uncertain['Value'], label='State Uncertain Transition', marker='x')

    # 添加图表标题和坐标轴标签
    # plt.title('Reward Change Over Steps in Different Environments')
    plt.xlabel('Step')
    plt.ylabel('Reward Value')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()





if __name__ == "__main__":
    # draw_all_state()
    # draw_obs_imperfect()
    # draw_abnormal_event()
    # draw_param_change()
    # draw_uncertain_action()
    draw_uncertain_transition()
