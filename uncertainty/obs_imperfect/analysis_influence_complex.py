"""分析复杂场景（有干预动作）下，影响信息重建的因素"""

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from algorithm.ppo_discrete_gpu import PPO_discrete_gpu
from utils.normalization import Normalization
from train_gpu import _config_args

def _eval_rebuild_effect(env, agent, state_norm, anal_idx, initial_region_ratio = 1.0, no_initial_region_ratio = 1.0):
    env.seed(3047)
    # 0.确定初始感染者撒在人数最多的100个区域上
    rand_idxs = torch.argsort(env.POP, dim=-1, descending=True)[:, :100]
    rand_idxs = rand_idxs.to(env.device)    # (env_cnt, 100)

    # 1.重置环境
    s = env.reset(rand_idxs)

    # 2.确定mask掉的区域
    initial_idx = torch.where(env.simRes[0, 0, :, env.E_undetected] > 0)[0]
    no_initial_idx = torch.where(env.simRes[0, 0, :, env.E_undetected] == 0)[0]

    report_mask = torch.zeros((env.env_count, env.ZONE_NUM), dtype=torch.bool, device=env.device)
    for i in range(env.env_count):
        num_report_init = int(len(initial_idx) * initial_region_ratio)
        report_init_idx = initial_idx[torch.randperm(len(initial_idx))[:num_report_init]]

        num_report_no_init = int(len(no_initial_idx) * no_initial_region_ratio)
        report_no_init_idx = no_initial_idx[torch.randperm(len(no_initial_idx))[:num_report_no_init]]

        report_idx = torch.cat([report_init_idx, report_no_init_idx])

        report_mask[i, report_idx] = True

    no_report_mask = ~report_mask
    env.set_mask(no_report_mask)

    # 3.达到anal_idx
    while env.day - 1 < anal_idx:
        # s = state_norm(s, update=False)
        # a = agent.evaluate(s)
        # s_, r, done, info = env.step(a)
        # s = s_
        a = torch.zeros((env.env_count, env.ZONE_NUM), device=env.device)
        env.step(a)

    # 4.分析重建效果
    POP = env.POP[0]
    OD = env.OD[0]
    OD_t = OD.transpose(0, 1)
    EI_idx = [env.E_undetected, env.E_detected, env.I_reported, env.I_undetected, env.I_detected]
    known_cnts, rs, p_values, mses, r2s = [], [], [], [], []
    for i in range(env.env_count):
        obs = env.history_local_obs[i, anal_idx, :, :2] * POP.unsqueeze(-1)   # (zone_num, 2)

        # u_p_test, u_p_quara = env._action_to_u(env.actions[i, anal_idx - 1, :])
        # obs_mask = u_p_test == 0
        obs_mask = no_report_mask[i, :]

        predict = torch.mm(OD_t, obs) / torch.mm(OD_t, POP.unsqueeze(-1))
        predict = torch.mm(OD, predict) * POP.unsqueeze(-1)
        predict = torch.clip(predict, min=0)

        # 数值校正
        numerical_ratio = obs[~obs_mask, :].sum(dim=0) / (predict[~obs_mask, :].sum(dim=0) + 1e-16)
        predict = predict * numerical_ratio.unsqueeze(0)

        y_rebuild = predict[obs_mask, 0].cpu().numpy()  # 评估curr_EI
        y_true = (env.simRes[i, anal_idx, :, EI_idx].sum(dim=-1))[obs_mask].cpu().numpy()

        # # 绘制x y 散点图
        # plt.plot(y_true, y_rebuild, 'o', color='blue')
        # plt.xlabel("True state")
        # plt.ylabel("Rebuild state")
        # plt.show()
        if len(y_rebuild) >= 2:
            r, p_value = pearsonr(y_true, y_rebuild)

            mse = mean_squared_error(y_true, y_rebuild)

            r2 = r2_score(y_true, y_rebuild)

        else:
            r, p_value = 1, 0
            mse = 0
            r2 = 1

        known_cnts.append((~obs_mask).float().sum())
        rs.append(r)
        p_values.append(p_value)
        mses.append(mse)
        r2s.append(r2)
    known_cnt = torch.tensor(known_cnts).float().mean()
    r, p_value = torch.tensor(rs).float().mean(), torch.tensor(p_values).float().mean()
    mse, r2 = torch.tensor(mses).float().mean(), torch.tensor(r2s).float().mean()

    return known_cnt, r, p_value, mse, r2

def analysis():
    # 定义环境
    args = _config_args()
    args.env_data_dir = '../../data/'
    args.use_obs_imperfect = True
    env = EpidemicModel(args, env_count=100)
    env.seed(3047)

    # 加载智能体
    args.model_idx = 12
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent.load(args.model_idx)
    # 状态归一化
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 收集分析数据
    # _eval_rebuild_effect(env, agent, state_norm, 3, 1, 1)

    anal_idxs = [1, 3, 5, 7, 10, 15, 20, 40, 80, 100]
    initial_region_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    no_initial_region_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # anal_idxs = [1, 3]
    # initial_region_ratios = [0.1, 1.0]
    # no_initial_region_ratios = [0.1, 1.0]
    # 创建一个 DataFrame 来存储分析结果
    results = []
    for anal_idx in anal_idxs:
        for irr in initial_region_ratios:
            for nirr in no_initial_region_ratios:
                known_cnt, r, p_value, mse, r2 = _eval_rebuild_effect(env, agent, state_norm, anal_idx, irr, nirr)

                # 将结果添加到列表中
                results.append({
                    "anal_idx": anal_idx,
                    "initial_region_ratio": irr,
                    "no_initial_region_ratio": nirr,
                    "known_cnt": known_cnt.item(),
                    "pearson_r": r.item(),
                    "p_value": p_value.item(),
                    "mse": mse.item(),
                    "r2": r2.item()
                })
                print("anal_idx:", anal_idx, "initial_region_ratio:", irr, "no_initial_region_ratio:", nirr)
    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(results)

    results_df.to_csv("analysis_influence_complex_no_action.csv", index=False)

def plot_results_df(results_df = None):
    if results_df is None:
        results_df = pd.read_csv("analysis_influence_complex.csv")
        # results_df = pd.read_csv("analysis_influence_complex_no_action.csv")

    # 对于 pearson_r 列，空填充为1
    results_df['pearson_r'].fillna(0, inplace=True)

    # 确保分析索引是唯一的
    anal_idxs = results_df['anal_idx'].unique()

    # 图1：热力图
    for anal_idx in anal_idxs:
        # 筛选特定 anal_idx 的数据
        subset_df = results_df[results_df['anal_idx'] == anal_idx]

        # 创建透视表，用于绘制热力图
        heatmap_data = subset_df.pivot(index='no_initial_region_ratio',
                                       columns='initial_region_ratio',
                                       values='pearson_r')

        # 绘制热力图
        plt.figure(figsize=(7, 5))
        plt.title(f"Heatmap for anal_idx = {anal_idx}", fontsize=14)
        plt.xlabel("Initial Region Ratio", fontsize=12)
        plt.ylabel("No Initial Region Ratio", fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        # 使用 imshow 绘制热力图
        c = plt.imshow(heatmap_data, origin='lower', aspect='auto', cmap='viridis', vmin=0.7, vmax=1.0)
        plt.colorbar(c, label="Pearson r")
        plt.xticks(ticks=range(len(heatmap_data.columns)), labels=heatmap_data.columns)
        plt.yticks(ticks=range(len(heatmap_data.index)), labels=heatmap_data.index)

        # 显示图像
        plt.tight_layout()
        plt.show()


    # 图2：pearson_r随着anal_idx的变化
    # 设置绘图
    plt.figure(figsize=(10, 6))

    # 定义颜色映射
    cmap = plt.get_cmap('YlGnBu')  # 从浅到深的颜色

    # 遍历不同的初始区域比例组合
    for irr in results_df['initial_region_ratio'].unique():
        for nirr in results_df['no_initial_region_ratio'].unique():
            # plt.figure(figsize=(7, 5))
            # 筛选当前组合下的数据
            subset_df = results_df[(results_df['initial_region_ratio'] == irr) &
                                   (results_df['no_initial_region_ratio'] == nirr)]

            # 根据 initial_region_ratio + no_initial_region_ratio 的和确定颜色
            color_value = irr + nirr
            color = cmap(color_value / (1.0 + 1.0))  # 归一化到 [0, 1]

            # 绘制曲线：x轴为 anal_idx，y轴为 pearson_r
            plt.plot(subset_df['anal_idx'], subset_df['pearson_r'],
                     label=f'IRR: {irr}, NIRR: {nirr}', color=color)
            # plt.title(f"Pearson r vs anal_idx for IRR: {irr}, NIRR: {nirr}")
            # plt.show()

    # 设置图表标题和轴标签
    plt.title("Pearson r vs anal_idx for different initial_region_ratio and no_initial_region_ratio combinations",
              fontsize=14)
    plt.xlabel("Analytical Index (anal_idx)", fontsize=12)
    plt.ylabel("Pearson r", fontsize=12)

    # 设置图例
    # plt.legend(title="Initial Region Ratios (IRR & NIRR)", loc="upper right", bbox_to_anchor=(1.1, 1))

    # 显示图像
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # analysis()

    plot_results_df()