import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import os

from uncertainty.obs_imperfect.main_rebuild import _get_history_padding
from utils.normalization import Normalization, RewardScaling
from algorithm.ppo_discrete_gpu import PPO_discrete_gpu
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
# from uncertainty.obs_imperfect.gru_model import RebuildGruModel, RebuildGru
from utils.general_functions import mask_params_to_str


def _evaluate_policy(args, env, agent, state_norm=None, evaluate_num=0, rebuild_method='ode_formula'):
    """
    评估模型：从初始状态开始，走完num_episodes条完整的轨迹
    """

    # 如果使用beta_change, 就需要重置beta_matrix
    if args.use_beta_change:
        env.dynamic_beta_change(type=args.test_beta_change_rule, std=0.2)
    s = env.reset()
    POP = env.POP
    OD = env.OD[0].unsqueeze(0).expand(env.env_count, -1, -1)
    OD_t = OD.transpose(1,2)
    done = False

    rebuild_states1 = torch.zeros((env.env_count, env.period + 1, env.ZONE_NUM, 2), device=env.device)  # 做记录
    rebuild_states2 = torch.zeros((env.env_count, env.period + 1, env.ZONE_NUM, 2), device=env.device)  # 为智能体服务，5天之前都是实际观测
    start_rebuild_day = 1 if draw_data_output_path is None or draw_data_output_path != "../../draw/地图对比" else 8

    # 定义两个重建评价指标
    predict_slice_effect = -1.0
    predict_total_effect = -1.0

    # 读取距离矩阵，为idw服务
    # 加载距离矩阵
    matrix_path = "../../data/sz/community_654/distance_matrix.npy"
    distance_matrix = np.load(matrix_path)
    distance_matrix = torch.from_numpy(distance_matrix).float().to(env.device)

    while not done:
        if args.use_rebuild:
            history_action = _get_history_padding(env.actions, env.day - 1, 7)
            obs = _get_history_padding(env.history_local_obs[:, :, :, :2], env.day, 7) * env.POP.unsqueeze(
                1).unsqueeze(-1)

            # a.获取未检测的区域
            u_p_test, u_p_quara = env._action_to_u(env.actions[:, env.day - 2, :])
            # obs_mask = obs[:, -1, :, 0] == 0
            obs_mask = u_p_test == 0

            """进行预测，也即信息重建"""
            if rebuild_method == 'ode_formula':
                """重建逻辑1，使用数学公式"""
                predict = _rebuild_ode_formula(OD_t=OD_t, OD=OD, POP=POP, obs=obs, obs_mask=obs_mask)
            elif rebuild_method == 'gnn_gru':
                """重建逻辑2，使用 args.trainer"""
                predict = args.trainer.predict(obs=obs, actions=history_action) # (env.env_count, env.ZONE_NUM, 2)
            elif rebuild_method == 'idw':
                """重建逻辑3，距离倒数"""
                predict = _rebuild_idw(obs=obs, obs_mask=obs_mask, distance_matrix=distance_matrix)
            else:
                raise ValueError("rebuild_method error. rebuild_method = ", rebuild_method)
            """重建后的处理"""
            # 方式1：直接用已知现存预测未知现存
            rebuild_states1[:, env.day - 1, :, 0] = predict[:, :, 0]    # 记录预估出来的现存
            rebuild_states1[:, env.day - 1, :, 1] = predict[:, :, 1]    # 记录预估出来的新增
            env.record_rebuild_state(rebuild_states1[:, env.day - 1, :, :].clone())

            # # 方式2：用预估新增估计未知现存（前一天的存留+今天的新增）
            # rebuild_states1[:, env.day - 1, :, 0] = rebuild_states1[:, env.day - 2, :, 0] * (1 - 1 / (3)) \
            #                                         + rebuild_states1[:, env.day - 1, :, 1]
            # rebuild_states1[:, env.day - 1, :, 0][~obs_mask] = obs[:, -1, :, 0][~obs_mask] # 有检测的区域改用实际观测

            # 相关系数计算
            # x = env.simState[:, :,[env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].mean(dim=0).sum(dim=-1)[obs_mask[0]]
            # y = rebuild_states1[:, env.day - 1, :, 0].mean(dim=0)[obs_mask[0]]
            # x2 = env.daily_new_E[:, env.day - 1, :].mean(dim=0)[obs_mask[0]]
            # y2 = rebuild_states1[:, env.day - 1, :, 1].mean(dim=0)[obs_mask[0]]
            x = env.simState[:, :,
                [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].mean(dim=0).sum(
                dim=-1)
            y = rebuild_states1[:, env.day - 1, :, 0].mean(dim=0)
            x2 = env.daily_new_E[:, env.day - 1, :].mean(dim=0)
            y2 = rebuild_states1[:, env.day - 1, :, 1].mean(dim=0)
            if env.day - 1 == 2: # 第2天记录一下预测效果
                predict_slice_effect = _evaluate_predict_effects(x2, y2)
            if env.day in range(2,10):
                _analysis_rebuild_effect(x, y, title=f"Day {env.day - 1} (curr_EI)")
                _analysis_rebuild_effect(x2, y2, title=f"Day {env.day - 1} (Eun)")

            if env.day >= start_rebuild_day:
                """在第5天才使用重建信息帮助决策"""
                rebuild_states2[:, env.day - 1, :, :] = rebuild_states1[:, env.day - 1, :, :]
            else:
                rebuild_states2[:, env.day - 1, :, :] = obs[:, -1, :, :]

            r_s = _get_history_padding(rebuild_states2 / POP.unsqueeze(1).unsqueeze(-1), env.day, env.WINDOW_SIZE)
            r_s = torch.cat((r_s, s[:,:,:,2:]), dim=-1)
            s = r_s
        if args.use_state_norm:
            s = state_norm(s, update=False)
        a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
        # if env.day < start_rebuild_day and draw_data_output_path == "../../draw/地图对比":
        #     a = torch.zeros_like(a)
        #     no_mask = a[0, :] == 0
        #     for region_idx in range(0, env.ZONE_NUM):
        #         if region_idx % 10 == 1:
        #             no_mask[region_idx] = True
        #         else:
        #             no_mask[region_idx] = False
        #     a[:, no_mask] = 3
        s_, r, done, info = env.step(a)

        s = s_

    if args.show_fig:
        env.render(title="eval:" + str(evaluate_num))
        plot_all_env(rebuild_states1, env)

        # for region_idx in range(0, 20):
        #     plot_one_region(rebuild_states, env, 0, region_idx)

    # 评价总体预测效果 TODO: 目前的rmse计算是各个环境平均后再计算，是否改为各个环境计算RMSE再平均？
    predict_total = rebuild_states1[:, :, :, 0].mean(dim=0).sum(dim=-1)
    target_total = (env.simRes[:, :, :,[env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]]
                    .sum(dim=[-2, -1]).mean(dim=0))
    predict_total_effect = _evaluate_predict_effects(predict_total, target_total)

    if not args.use_rebuild:
        # 如果没有信息重建，就用观测和目标进行评估
        predict_slice = (env.history_local_obs[:, 2, :, 1] * env.POP).mean(dim=0)
        total_slice = (env.daily_new_E[:, 2, :].mean(dim=0))
        predict_slice_effect = _evaluate_predict_effects(predict_slice, total_slice)

        predict_total = (env.history_local_obs[:, :, :, 0] * env.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)
        predict_total_effect = _evaluate_predict_effects(predict_total, target_total)

    info["predict_slice_effect"] = predict_slice_effect
    info["predict_total_effect"] = predict_total_effect

    info["missing_rate"] = env.get_observation_missing_rate()

    return info

def _evaluate_predict_effects(predict: torch.Tensor, target: torch.Tensor) -> float:
    # 使用 RMSE 评估预测效果
    # 计算均方误差（MSE），再开平方根得到 RMSE
    mse = torch.mean((predict - target) ** 2)
    rmse = torch.sqrt(mse)

    # 返回 Python 浮点数（而非张量）
    return rmse.item()

def _rebuild_ode_formula(OD_t, OD, POP, obs, obs_mask):
    # b. 公式重建逻辑
    predict = torch.bmm(OD_t, obs[:, -1, :, :]) / torch.bmm(OD_t, POP.unsqueeze(-1))  # (env.env_count, env.ZONE_NUM, 2)
    predict = torch.bmm(OD, predict) * POP.unsqueeze(-1)
    predict = torch.clip(predict, min=0)
    # c. 数值修正
    numerical_ratio = [(obs[i, -1, ~obs_mask[i], :]).sum(dim=0) / (predict[i, ~obs_mask[i], :].sum(dim=0) + 1e-16)
                       for i in range(obs.shape[0])]
    numerical_ratio = torch.stack(numerical_ratio).unsqueeze(1)
    predict = predict * numerical_ratio
    predict[~obs_mask] = obs[:, -1, :, :][~obs_mask]  # 对于有观测的，用真实观测值回填。
    predict = torch.max(predict, obs[:, -1, :, :])  # 重建和观测取较大的，这个是否要保留？

    return predict



def _rebuild_idw(obs, obs_mask, distance_matrix, power=2.0, eps=1e-8):
    """
        向量化IDW算法填充缺失值（obs_mask=True的区域）
    """
    obs = obs[:, -1, :, :].clone()
    predict = obs.clone()

    # 对每个批次和每个特征独立处理
    for batch in range(predict.shape[0]):
        for feature in range(predict.shape[2]):
            current_obs = obs[batch, :, feature]

            current_mask = obs_mask[batch, :]

            missing_indices = torch.where(current_mask)[0]

            known_indices = torch.where(~current_mask)[0]

            if len(known_indices) == 0:
                continue

            # --- 向量化核心计算 ---
            # 提取缺失点与已知点的距离 (n_missing, n_known)
            dists = distance_matrix[missing_indices][:, known_indices]  # (n_missing, n_known)

            # 计算权重 (n_missing, n_known)
            weights = 1.0 / (dists.pow(power) + eps)

            # 已知值 (n_known,)
            known_values = obs[batch, known_indices, feature]

            # 加权求和 (n_missing,)
            weighted_sum = (weights * known_values).sum(dim=1)
            sum_weights = weights.sum(dim=1)

            # 填充结果
            predict[batch, missing_indices, feature] = weighted_sum / sum_weights


    return predict


def _analysis_rebuild_effect(x, y, title ="Day 0 (curr_EI)"):
    """x为真实，y为预测"""
    if len(x) < 2 or y.sum() == 0 or x.sum() == 0:
        r, p_value = 0, 0
        r2 = 0
    else:
        r, p_value = pearsonr(x.cpu().numpy(), y.cpu().numpy())
        r2 = r2_score(x.cpu().numpy(), y.cpu().numpy())
        # # 绘制x y 散点图
        # plt.plot(x.cpu().numpy(), y.cpu().numpy(), 'o', color='blue')
        # plt.xlabel("True state")
        # plt.ylabel("Rebuild state")
        # plt.title(f"{title} (pearson:{r:.4f}, R2:{r2:.4f})")
        # plt.show()
    print("==============", title, "==============")
    print(len(x))
    print(f"Pearson Correlation Coefficient: {r:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"R2: {r2:.4f}")


def plot_all_env(rebuild_states, env):

    # 对照1
    first_vector_1 = rebuild_states[:, :, :, 0].sum(dim=2).mean(dim=0)  # shape (120,)
    first_vector_2 = env.simRes[:, :, :,
                     [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(
        dim=[-2, -1]).mean(dim=0)
    first_vector_3 = (env.history_local_obs[:, :, :, 0] * env.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)

    # 对照2
    second_vector_1 = rebuild_states[:, :, :, 1].sum(dim=2).mean(dim=0)  # shape (120,)
    # second_vector_2 = env.simRes[:, :, :, env.E_undetected].sum(dim=-1).mean(dim=0)  # shape (120,) todo:这个是否替换
    second_vector_2 = env.daily_new_E.sum(dim=-1).mean(dim=0)  # shape (120,) 这是新增的预测目标
    second_vector_3 = (env.history_local_obs[:, :, :, 1] * env.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)

    _plot_6_compare(first_vector_1, first_vector_2, first_vector_3, second_vector_1, second_vector_2, second_vector_3)


def _plot_6_compare(first_vector_1, first_vector_2, first_vector_3, second_vector_1, second_vector_2, second_vector_3, region_idx=None):
    # 绘制图形
    plt.figure(figsize=(10, 8))
    # 第一个对照（第一对曲线）
    plt.subplot(2, 1, 1)
    plt.plot(first_vector_1.cpu().numpy(), label="Rebuild States - E/I", color='blue')
    plt.plot(first_vector_2.cpu().numpy(), label="True - E/I", color='green')
    plt.plot(first_vector_3.cpu().numpy(), label="Obs - E/I", color='orange')
    title1 = "Comparison 1: Rebuild States vs SimRes"
    if region_idx is not None:
        title1 += f" (Region {region_idx})"
    plt.title(title1)
    plt.xlabel("Index (Time Step)")
    plt.ylabel("Value")
    plt.legend()
    # 第二个对照（第二对曲线）
    plt.subplot(2, 1, 2)
    plt.plot(second_vector_1.cpu().numpy(), label="Rebuild States - E/I", color='blue')
    plt.plot(second_vector_2.cpu().numpy(), label="True New EI", color='green')
    plt.plot(second_vector_3.cpu().numpy(), label="Obs New EI", color='orange')
    # plt.plot(second_vector_4.cpu().numpy(), label="Daily Detect EI", color='blue')
    title2 = "Comparison 2: Rebuild States vs Daily New EI"
    if region_idx is not None:
        title2 += f" (Region {region_idx})"
    plt.title(title2)
    plt.xlabel("Index (Time Step)")
    plt.ylabel("Value")
    plt.legend()
    # 显示图形
    plt.tight_layout()
    plt.show()

def _extract_state_from_env(env: EpidemicModel, path="../../draw/监测预警建模竞赛/perfect_report"):
    # 从env提取self.simRes = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 9), device=self.device)
    simRes = env.simRes.cpu().numpy()
    np.save(path + '/simRes.npy', simRes)
    print("simRes shape: ", simRes.shape)

    # 抽取新增 env.daily_new_E
    daily_new_E = env.daily_new_E.cpu().numpy()
    np.save(path + '/daily_new_E.npy', daily_new_E)
    print("daily_new_E shape: ", daily_new_E.shape)

    # 提取obs
    obs = env.extract_observe().cpu().numpy()
    np.save(path + '/obs.npy', obs)
    print("obs shape: ", obs.shape)

    # 提取动作：
    actions = env.actions.cpu().numpy()
    np.save(path + '/actions.npy', actions)
    print("actions shape: ", actions.shape)

    # 提取每日隔离人数
    daily_quara_num = env.extract_daily_quara_num().cpu().numpy()
    np.save(path + '/daily_quara_num.npy', daily_quara_num)
    print("daily_quara_num shape: ", daily_quara_num.shape)

    # 提取rebuild_state
    rebuild_states = env.extract_rebuild_state().cpu().numpy()
    np.save(path + '/rebuild_states.npy', rebuild_states)
    print("rebuild_states shape: ", rebuild_states.shape)


def main():
    from train_gpu import _config_args
    args = _config_args()
    # args.device_name = 'cpu'
    args.daily_imported_cases = 1
    args.show_fig = True

    # 低R0实验
    args.ODE_beta = 0.4
    args.R0 = 'low'

    # 验证检测资源效率函数的指数变化影响
    args.detection_efficiency_exp_param = 0.65

    # 指定要加载的智能体
    args.max_train_steps = 2400 * 20
    args.model_idx = 18
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent_original_dir = agent.directory
    agent.load(args.model_idx)
    # 状态归一化
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 初始化一个观测完全的环境
    args.env_data_dir = '../../data/'
    env_obs_perfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_perfect.seed()
    # 不使用信息重建
    args.use_rebuild = False
    infos = []
    info_certain = _evaluate_policy(args, env_obs_perfect, agent, state_norm=state_norm)
    info_certain['name'] = 'Certain'
    infos.append(info_certain)

    # 初始化一个观测不完全的环境
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    # 不使用信息重建
    args.use_rebuild = False
    info_uncertain = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm)
    info_uncertain['name'] = 'Uncertain'
    infos.append(info_uncertain)

    # 使用信息重建
    args.use_rebuild = True

    # 1.机理
    # 1.1. ODE
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='ode_formula')
    info_rebuild['name'] = 'ode_formula'
    infos.append(info_rebuild)

    # 2.数据驱动
    # 2.1.idw
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='idw')
    info_rebuild['name'] = 'idw'
    infos.append(info_rebuild)

    # 2.2. gnn-gru
    # 加载模型
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2
    # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
    env_temp = EpidemicModel(args, env_count=1)
    env_temp.POP = None
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_ordinary.pth'))
    args.trainer = trainer
    # 测试
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    info_rebuild['name'] = 'gnn_gru_ordinary'
    infos.append(info_rebuild)

    # 3.机理+数据驱动
    # 3.1. agent数据集
    # 加载模型
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2
    # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
    env_temp = EpidemicModel(args, env_count=1)
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_agent.pth'))
    args.trainer = trainer
    # 测试
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    info_rebuild['name'] = 'gnn_gru_agent'
    infos.append(info_rebuild)

    # 3.2. random数据集
    # 加载模型
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2
    # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
    env_temp = EpidemicModel(args, env_count=1)
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_random.pth'))
    args.trainer = trainer
    # 测试
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    info_rebuild['name'] = 'gnn_gru_random'
    infos.append(info_rebuild)

    # 3.3. 包含动作
    # 加载模型
    from uncertainty.obs_imperfect.gru_gnn_model_v2 import RebuildGruGNNModel, RebuildGruGNN
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2
    # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
    env_temp = EpidemicModel(args, env_count=1)
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name, node_output_size=48)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len, env=env_temp)
    trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_with_action.pth'))
    args.trainer = trainer
    # 测试
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    info_rebuild['name'] = 'gnn_gru_with_action'
    infos.append(info_rebuild)

    # 4. e2e
    # 4.1. 在不完全观测下直接训练的rl
    # 加载智能体
    args.use_rebuild = False
    args.rl_type = 'uncertainty'
    args.predictor_type = 'none'
    args.use_obs_imperfect = True  # 表示加载不确定环境下协同训练出的智能体
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent.load(args.model_idx)
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')
    # 测试
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm)
    info_rebuild['name'] = 'rl_train_in_imperfect_obs'
    infos.append(info_rebuild)

    # 4.2. e2e
    args.use_rebuild = True
    for p_type in ['rebuild_gru_gnn_model_with_action', 'rebuild_gru_gnn_model_no_action']:
        # 加载智能体
        args.model_idx = 19 if args.R0 == 'high' else 18
        args.rl_type = 'uncertainty'
        args.predictor_type = p_type
        args.use_obs_imperfect = True # 表示加载不确定环境下协同训练出的智能体
        agent = PPO_discrete_gpu(args)
        agent.directory = '../../' + agent.directory
        agent.load(args.model_idx)
        if args.use_state_norm:
            state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
            state_norm.load(agent.directory, filename='state_norm.pth')
        # 初始化模型 gnn-gru-with-action-edge2edge （agent stateNormal 都要重新加载）
        from uncertainty.obs_imperfect.generic_predictor import GenericPredictor
        args.env_count = repeats
        predictor = GenericPredictor(args, agent.directory)
        predictor.load(args.model_idx)
        args.trainer = predictor.get_trainer()

        # 测试
        args.use_obs_imperfect = True
        env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
        env_obs_imperfect.seed()
        info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
        info_rebuild['name'] = 'gnn_gru_e2e' + p_type
        infos.append(info_rebuild)


    # 导出数据表格
    import pandas as pd

    # 预处理：对每个字典的特定字段进行计算
    processed_data = [
        {
            "name": info["name"],
            "predict_total_effect": info["predict_total_effect"],
            "predict_slice_effect": info["predict_slice_effect"],
            "ep_r": info["ep_r"].mean().item(),
            "total_infections": info["total_infections"].mean().item(),
            "total_test_num": info["total_test_num"].mean().item(),
            "total_quara_num": info["total_quara_num"].mean().item(),
            "total_control_cost": info["total_control_cost"].mean().item(),
            "score": info["score"].mean().item(),  # 计算均值并转为 Python 标量
            "score_avg": info["score_avg"].item(),
            "avg_zone_score": info["avg_zone_score"].mean().item(),
        }
        for info in infos
    ]

    # 生成 DataFrame
    df = pd.DataFrame(processed_data)

    # 输出为 Excel
    excel_path = agent_original_dir + '/' + f'processed_scores_{repeats}_detection_{args.detection_efficiency_exp_param}.xlsx'
    csv_file_path = agent_original_dir + '/' + f'processed_scores_{repeats}_detection_{args.detection_efficiency_exp_param}.csv'
    df.to_csv(csv_file_path, index=False)
    print("DataFrame has been exported to CSV: ", csv_file_path)
    df.to_excel(excel_path, index=False)
    print("DataFrame has been exported to Excel: ", excel_path)

def extract_data_for_partial_day():
    """
    从env中提取数据用于绘图

    绘图类型：
        地图对比：使用ODE机理重建作为“rebuild_info”来源，更好控制到第7/8天才开始重建
        监测预警建模竞赛：偏重于时间展示，因而使用`3.1. agent数据集`作为“rebuild_info”来源
    """
    # draw_data_output_path = "../../draw/地图对比"
    draw_data_output_path = "../../draw/监测预警建模竞赛"
    repeats = 2

    from train_gpu import _config_args
    args = _config_args()
    args.daily_imported_cases = 1
    args.show_fig = True

    # 指定要加载的智能体
    args.max_train_steps = 2400 * 20
    args.model_idx = 18
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent_original_dir = agent.directory
    agent.load(args.model_idx)
    # 状态归一化
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 初始化一个观测完全的环境
    args.env_data_dir = '../../data/'
    env_obs_perfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_perfect.seed()
    # 不使用信息重建
    args.use_rebuild = False
    infos = []
    info_certain = _evaluate_policy(args, env_obs_perfect, agent, state_norm=state_norm)
    info_certain['name'] = 'Certain'
    infos.append(info_certain)
    _extract_state_from_env(env_obs_perfect, draw_data_output_path + "/perfect_report")

    # 初始化一个观测不完全的环境
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    # 设置mask
    report_mask = torch.zeros((env_obs_imperfect.env_count, env_obs_imperfect.ZONE_NUM), dtype=torch.bool, device=env_obs_imperfect.device)
    for i in range(env_obs_imperfect.env_count):
        report_idx = []
        for region_idx in range(env_obs_imperfect.ZONE_NUM):
            if region_idx % 3 == 0:
                report_idx.append(region_idx)

        report_mask[i, report_idx] = True
    env_obs_imperfect.set_mask(~report_mask)
    # 不使用信息重建
    args.use_rebuild = False
    info_uncertain = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm)
    info_uncertain['name'] = 'Uncertain'
    infos.append(info_uncertain)
    _extract_state_from_env(env_obs_imperfect, draw_data_output_path + "/partial_observable")

    # 使用信息重建
    args.use_rebuild = True

    # 1.机理
    # 1.1. ODE
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    env_obs_imperfect.set_mask(~report_mask)
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='ode_formula')
    info_rebuild['name'] = 'ode_formula'
    infos.append(info_rebuild)
    _extract_state_from_env(env_obs_imperfect, draw_data_output_path + "/rebuild_info")

    # # 3.1. agent数据集
    # # 加载模型
    # from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
    # seq_len = 7
    # hidden_size = 128
    # output_size = 2
    # num_layers = 2
    # # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
    # env_temp = EpidemicModel(args, env_count=1)
    # model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
    #                            env=env_temp, device_name=args.device_name)
    # trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    # trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_agent.pth'))
    # args.trainer = trainer
    # # 测试
    # args.use_obs_imperfect = True
    # env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    # env_obs_imperfect.seed()
    # env_obs_imperfect.set_mask(~report_mask)
    # info_rebuild = evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    # info_rebuild['name'] = 'gnn_gru_agent'
    # infos.append(info_rebuild)
    # extract_state_from_env(env_obs_imperfect, draw_data_output_path + "/rebuild_info")

def _analysis_observations_missing_rate(args, missing_rate_table_file_path=""):
    """
    分析观测缺失率与score的关系
    """
    repeats = 20

    # 指定要加载的智能体
    args.max_train_steps = 2400 * 20
    args.model_idx = 18
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent_original_dir = agent.directory
    agent.load(args.model_idx)
    # 状态归一化
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 用一个表格记录结果，用于后续绘图，列名为：mask_rate_up, mask_rate_down, mask_duration_up, mask_duration_down, name, missing_rate, score
    rows = []

    random_report_rate_down = random.Random(3047)
    seeds = [int(random_report_rate_down.uniform(10, 10000)) for _ in range(repeats)]
    for mask_rate_down in [0, 0.2, 0.4, 0.6, 0.8]:
        mask_rate_up = mask_rate_down + 0.2
        args.mask_rate_up = mask_rate_up
        args.mask_rate_down = mask_rate_down
        for mask_duration_down in [14, 28, 42, 56, 70, 84, 98]:
            mask_duration_up = mask_duration_down + 14
            args.mask_duration_up = mask_duration_up
            args.mask_duration_down = mask_duration_down
            mask_str = mask_params_to_str(args)

            # for seed in [3047, 42, 69, 420, 666]:
            for seed in seeds:
                # 初始化一个观测完全的环境
                args.env_data_dir = '../../data/'
                args.use_obs_imperfect = False
                env_obs_perfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_perfect.seed(seed)
                # 不使用信息重建
                args.use_rebuild = False
                infos = []
                info_certain = _evaluate_policy(args, env_obs_perfect, agent, state_norm=state_norm)
                info_certain['name'] = 'Certain'
                infos.append(info_certain)

                # 初始化一个观测不完全的环境
                args.use_obs_imperfect = True
                args.obs_imperfect_down = 0.1
                args.obs_imperfect_up = 1
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                # 不使用信息重建
                args.use_rebuild = False
                info_uncertain = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm)
                info_uncertain['name'] = 'Uncertain'
                infos.append(info_uncertain)

                # 使用信息重建
                args.use_rebuild = True

                # 1.机理
                # 1.1. ODE
                args.use_obs_imperfect = True
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='ode_formula')
                info_rebuild['name'] = 'ode_formula'
                infos.append(info_rebuild)

                # 2.数据驱动
                # 2.1.idw
                args.use_obs_imperfect = True
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='idw')
                info_rebuild['name'] = 'idw'
                infos.append(info_rebuild)

                # 2.2. gnn-gru
                # 加载模型
                from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
                seq_len = 7
                hidden_size = 128
                output_size = 2
                num_layers = 2
                # 初始化一个临时环境，使用其 OD POP数据，以及action_to_u方法
                env_temp = EpidemicModel(args, env_count=1)
                env_temp.POP = None
                model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                                           env=env_temp, device_name=args.device_name)
                trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
                trainer.load(os.path.join(agent.directory, f'rebuild_gru_gnn_model_v1_ordinary.pth'))
                args.trainer = trainer
                # 测试
                args.use_obs_imperfect = True
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
                info_rebuild['name'] = 'gnn_gru_ordinary'
                infos.append(info_rebuild)

                # 3.1. agent数据集
                # 加载模型
                from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
                seq_len = 7
                hidden_size = 128
                output_size = 2
                num_layers = 2
                # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
                env_temp = EpidemicModel(args, env_count=1)
                model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size,
                                           gru_num_layers=num_layers,
                                           env=env_temp, device_name=args.device_name)
                trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
                trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_agent.pth'))
                args.trainer = trainer
                # 测试
                args.use_obs_imperfect = True
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm,
                                                rebuild_method='gnn_gru')
                info_rebuild['name'] = 'gnn_gru_agent'
                infos.append(info_rebuild)


                missing_rate_avg = info_rebuild['missing_rate'].mean().item()
                print(f"mask_rate_down: {mask_rate_down:.2f}, "
                      f"mask_rate_up: {mask_rate_up:.2f}, "
                      f"mask_duration_down: {mask_duration_down:.2f}, "
                      f"mask_duration_up: {mask_duration_up:.2f}, "
                      f"missing_rate_avg: {missing_rate_avg:.4f}")

                # 从infos中提取每种name的 missing_rate 和 score 对应关系
                for info in infos:
                    for i in range(info['missing_rate'].shape[0]):
                        missing_rate = info["missing_rate"][i].item()
                        if info['name'] == 'Certain':
                            missing_rate = info_uncertain["missing_rate"][i].item()
                        row = {
                            "mask_rate_up": float(args.mask_rate_up),
                            "mask_rate_down": float(args.mask_rate_down),
                            "mask_duration_up": int(args.mask_duration_up),
                            "mask_duration_down": int(args.mask_duration_down),
                            "name": info["name"],
                            "missing_rate": missing_rate,
                            "score": info["score"][i].item(),
                            "predict_total_effect": info["predict_total_effect"],
                        }
                        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_excel(missing_rate_table_file_path, index=False)

def analysis_observations_missing_rate():
    from train_gpu import _config_args

    args = _config_args()
    # args.device_name = 'cpu'
    args.daily_imported_cases = 1
    args.show_fig = False

    missing_rate_table_file_path = f"../../draw/缺失率影响分析/missing_rate_table_{args.R0}.xlsx"
    _analysis_observations_missing_rate(args, missing_rate_table_file_path)

    # 低R0实验
    args.ODE_beta = 0.4
    args.R0 = 'low'
    missing_rate_table_file_path = f"../../draw/缺失率影响分析/missing_rate_table_{args.R0}.xlsx"
    _analysis_observations_missing_rate(args, missing_rate_table_file_path)


repeats = 100
draw_data_output_path = None
if __name__ == '__main__':
    main()
    # extract_data_for_partial_day()

    # analysis_observations_missing_rate()

