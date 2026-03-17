"""只拿相邻两天的数据进行分析"""

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def _get_data():
    """拿到相邻两天的数据，返回一个env"""
    from config import args
    args.env_data_dir = '../../data/'

    args.simulate_scale = 'community'
    args.zone_num = 654
    args.use_obs_imperfect = False

    env = EpidemicModel(args, env_count=2)
    env.seed(3047)

    # 随机种子控制
    # # 方式一，随机化
    # random.seed(3074)
    # rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    # rand_idxs = torch.tensor(rand_list, device=env.device)
    # # 将单个环境的索引复制到每个批次
    # rand_idxs = rand_idxs.repeat(env.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组

    # 方式二，选择人数最多的100个区域
    rand_idxs = torch.argsort(env.POP, dim=-1, descending=True)[:, :100]
    rand_idxs = rand_idxs.to(env.device)    # (env_count, 100)

    env.reset(rand_idxs)

    while True:
        action = torch.ones(env.env_count, env.ZONE_NUM).to(env.device)
        action = action * 0
        s_, r, done, info = env.step(action=action)
        if done:
            break
    env.render()
    return env


def _eval_rebuild_effect(EI_curr, EI_next, OD, POP, infection_region_ratio = 1.0, no_infection_region_ratio = 1.0):
    """在已知区域占比、对爆发点区域的已知程度 的影响下，使用信息重建，评估重建效果（相关系数、绝对值差）"""
    OD_t = OD.transpose(0, 1)

    seed = 3047
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 获取已感染和未感染的索引
    infected_idx = torch.where(EI_curr > 0)[0]  # 已感染区域索引
    non_infected_idx = torch.where(EI_curr == 0)[0]  # 未感染区域索引

    known_cnts, rs, p_values, mses, r2s = [], [], [], [], []
    for _ in range(100):
        # 从已感染区域中随机选择一部分作为已知区域
        num_known_infected = int(len(infected_idx) * infection_region_ratio)
        known_infected_idx = infected_idx[torch.randperm(len(infected_idx))[:num_known_infected]]

        # 从未感染区域中随机选择一部分作为已知区域
        num_known_non_infected = int(len(non_infected_idx) * no_infection_region_ratio)
        known_non_infected_idx = non_infected_idx[torch.randperm(len(non_infected_idx))[:num_known_non_infected]]

        # 合并已知区域索引
        known_idx = torch.cat([known_infected_idx, known_non_infected_idx])

        # 构建已知区域信息
        masked_EI_curr = torch.zeros_like(EI_curr)
        masked_EI_curr[known_idx] = EI_curr[known_idx]

        # 预测
        predict = torch.mm(OD_t, masked_EI_curr.unsqueeze(1)) / torch.mm(OD_t, POP.unsqueeze(1))
        predict = torch.mm(OD, predict)[:, 0] * POP

        # 修正
        predict = predict * (EI_curr[known_idx].sum() / predict[known_idx].sum() + 1e-12)

        eval_idx = torch.full(EI_curr.shape, True, dtype=torch.bool)
        eval_idx[known_idx] = False # 已知区域不评估
        y_rebuild = predict[eval_idx].cpu().numpy()
        y_true = EI_next[eval_idx].cpu().numpy()
        # # 绘制x y 散点图
        # plt.plot(y_true, y_rebuild, 'o', color='blue')
        # plt.xlabel("True state")
        # plt.ylabel("Rebuild state")
        # plt.show()
        if len(y_rebuild) >= 2:
            r, p_value = pearsonr(y_true, y_rebuild)

            mse = mean_squared_error(y_true, y_rebuild)

            r2 = r2_score(y_true, y_rebuild)
            # r2 = r2_score(y_rebuild, y_true)
        else:
            r, p_value = 1, 0
            mse = 0
            r2 = 1

        known_cnts.append(len(known_idx))
        rs.append(r)
        p_values.append(p_value)
        mses.append(mse)
        r2s.append(r2)
    known_cnt = torch.tensor(known_cnts).float().mean()
    r, p_value = torch.tensor(rs).float().mean(), torch.tensor(p_values).float().mean()
    mse, r2 = torch.tensor(mses).float().mean(), torch.tensor(r2s).float().mean()
    # print(f"Known region num: {known_cnt}, ratio: {known_cnt / len(EI_curr) : .4f}")
    # print(f"Pearson Correlation Coefficient: {r:.4f}")
    # print(f"P-value: {p_value:.4e}")
    # print(f"Mean Squared Error: {mse:.4f}")
    # print(f"R^2 Score: {r2:.4f}")


    return known_cnt, r, p_value, mse, r2


def plot_results_df(results_df = None):
    if results_df is None:
        results_df = pd.read_csv("analysis_influence_simple.csv")
    if not all(col in results_df.columns for col in ["anal_idx", "infection_region_ratio", "pearson_r", "r2"]):
        raise ValueError("results_df 缺少必要的列：'anal_idx', 'infection_region_ratio', 'pearson_r', 'r2'")

        # 获取唯一的 anal_idx 值（即不同的天数）
    anal_idxs = sorted(results_df["anal_idx"].unique())

    # 生成颜色从浅到深的渐变
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(anal_idxs)))

    # 绘制 Pearson Correlation 图
    plt.figure(figsize=(10, 6))
    for idx, color in zip(anal_idxs, colors):
        # 筛选出当前 anal_idx 的数据
        subset = results_df[results_df["anal_idx"] == idx]

        # 绘制曲线
        plt.plot(
            subset["infection_region_ratio"],
            subset["pearson_r"],
            label=f"Day {idx}",
            color=color
        )

    # 添加标题和标签
    plt.title("Pearson Correlation")
    plt.xlabel("Infection Region Ratio (known)")
    plt.ylabel("Pearson Correlation (r)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # 绘制 R^2 图
    plt.figure(figsize=(10, 6))
    for idx, color in zip(anal_idxs, colors):
        # 筛选出当前 anal_idx 的数据
        subset = results_df[results_df["anal_idx"] == idx]

        # 绘制曲线
        plt.plot(
            subset["infection_region_ratio"],
            subset["r2"],
            label=f"Day {idx}",
            color=color
        )

    # 添加标题和标签
    plt.title("R^2 Score")
    plt.xlabel("Infection Region Ratio (known)")
    plt.ylabel("R^2 Score")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def analysis():
    """进行分析"""

    # 1.提取数据（无干预动作影响）
    env = _get_data()
    anal_idx = 20
    EI_idx = [env.E_undetected, env.E_detected, env.I_reported, env.I_undetected, env.I_detected]
    EI_curr = env.simRes[0, anal_idx + 1, :, EI_idx].sum(dim=-1)
    EI_next = env.simRes[0, anal_idx + 1, :, EI_idx].sum(dim=-1)
    POP = env.POP[0]
    OD = env.OD[0]

    # 2.评估信息已知程度对重建的影响
    _eval_rebuild_effect(EI_curr, EI_next, OD, POP, 0.1, 0.0)

    anal_idxs = [1, 5, 10, 20, 50, 80, 110]
    infection_region_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # 创建一个 DataFrame 来存储分析结果
    results = []
    for anal_idx in anal_idxs:
        EI_curr = env.simRes[0, anal_idx, :, EI_idx].sum(dim=-1)
        EI_next = env.simRes[0, anal_idx, :, EI_idx].sum(dim=-1)
        for irr in infection_region_ratios:
            known_cnt, r, p_value, mse, r2 = _eval_rebuild_effect(EI_curr, EI_next, OD, POP, irr, 0.0)

            # 将结果添加到列表中
            results.append({
                "anal_idx": anal_idx,
                "infection_region_ratio": irr,
                "known_cnt": known_cnt.item(),
                "pearson_r": r.item(),
                "p_value": p_value.item(),
                "mse": mse.item(),
                "r2": r2.item()
            })

    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(results)

    results_df.to_csv("analysis_influence_simple.csv", index=False)
    
    # 绘图
    plot_results_df(results_df)

if __name__ == '__main__':
    # analysis()

    plot_results_df()