import glob
import os

import numpy as np

from algorithm.ppo_discrete_gpu import PPO_discrete_gpu
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
from train_gpu import _config_args
import torch
import matplotlib.pyplot as plt

from utils.normalization import Normalization


def collect_dataset(args, repeat_times=1, env_count=100, agent=None, state_norm=None):
    """从环境中生成数据集"""
    env = EpidemicModel(args, env_count=env_count)
    env.seed()
    device = env.device

    imperfect_obs_dataset = []
    history_action_dataset = []
    true_state_dataset = []

    for _ in range(repeat_times):
        s = env.reset()
        done = False
        while not done:
            # a = torch.zeros((env.env_count, env.ZONE_NUM), device=device)
            # 随机选择动作
            if agent is None:
                a = torch.randint(0, env.action_dim, (env.env_count, env.ZONE_NUM), device=device)
            else:
                # 使用ppo-agent 选择动作
                if state_norm is not None:
                    s = state_norm(s, update=False)
                # a, _ = agent.choose_action(s) # choose_action是按照概率进行采样,具有随机性
                a = agent.evaluate(s)
            s_, r, done, _ = env.step(a)
            s = s_
        env.render(f"数据收集后的绘图，repeat times: {repeat_times}")
        # 不完全观测
        # 从history_local_obs计算不完全观测信息（现存，新增）
        # imperfect_obs = env.history_local_obs[:, :, :, :2] * env.POP.unsqueeze(1).unsqueeze(-1) # (env_count, period + 1, ZONE_NUM, 2)
        imperfect_obs = env.extract_observe() # (env_count, period + 1, ZONE_NUM, 2)
        history_action = env.actions # (env_count, period + 1, ZONE_NUM)
        # 真实（现存，新增）
        curr_EI = env.simRes[:, :, :, [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(dim=-1)
        # new_EI = env.simRes[:, :, :, env.E_undetected]
        new_EI = env.daily_new_E    # (env_count, period + 1, ZONE_NUM)
        true_state = torch.stack([curr_EI, new_EI], dim=-1) # (env_count, period + 1, ZONE_NUM, 2)

        imperfect_obs_dataset.append(imperfect_obs)
        history_action_dataset.append(history_action)
        true_state_dataset.append(true_state)


    imperfect_obs_dataset = torch.cat(imperfect_obs_dataset, dim=0) # (total_episodes, period + 1, ZONE_NUM)
    history_action_dataset = torch.cat(history_action_dataset, dim=0) # (total_episodes, period + 1, ZONE_NUM)
    true_state_dataset = torch.cat(true_state_dataset, dim=0) # (total_episodes, period + 1, ZONE_NUM)

    # plot_all_env(imperfect_obs_dataset, true_state_dataset)
    #
    # for i in range(20):
    #     plot_one_region(imperfect_obs_dataset, true_state_dataset, region_idx=i)


    dataset = {
        'imperfect_obs': imperfect_obs_dataset,
        'history_action': history_action_dataset,
        'true_state': true_state_dataset,
    }

    print("数据集已生成：\n imperfect_obs:", imperfect_obs_dataset.shape,
          "\n history_action:", history_action_dataset.shape,
          "\n true_state:", true_state_dataset.shape)
    print("数据集中的：\n", "观测：实际数值（未除以POP） \n", "动作：动作下标 \n", "真实状态：预测目标，实际数值（未除以POP） ")

    return dataset

def plot_all_env(imperfect_obs_dataset, true_state_dataset):
    # 看一下收集的数据
    # 对照1
    first_vector_1 = imperfect_obs_dataset[:, :, :, 0].sum(dim=2).mean(dim=0)  # shape (120,)
    first_vector_2 = true_state_dataset[:, :, :, 0].sum(dim=2).mean(dim=0)

    # 对照2
    second_vector_1 = imperfect_obs_dataset[:, :, :, 1].sum(dim=2).mean(dim=0)  # shape (120,)
    second_vector_2 = true_state_dataset[:, :, :, 1].sum(dim=2).mean(dim=0)

    # 绘制图形
    plot_4_compare(first_vector_1, first_vector_2, second_vector_1, second_vector_2)

def plot_one_region(imperfect_obs_dataset, true_state_dataset, region_idx=0):
    idx = 0
    # 对照1
    first_vector_1 = imperfect_obs_dataset[idx, :, region_idx, 0]  # shape (120,)
    first_vector_2 = true_state_dataset[idx, :, region_idx, 0]

    # 对照2
    second_vector_1 = imperfect_obs_dataset[idx, :, region_idx, 0]  # shape (120,)
    second_vector_2 = true_state_dataset[idx, :, region_idx, 0]

    # 绘制图形
    plot_4_compare(first_vector_1, first_vector_2, second_vector_1, second_vector_2)




def plot_4_compare(first_vector_1, first_vector_2, second_vector_1, second_vector_2):
    plt.figure(figsize=(10, 8))
    # 第一个对照（第一对曲线）
    plt.subplot(2, 1, 1)
    plt.plot(first_vector_1.cpu().numpy(), label="obs States - E/I (0)", color='blue')
    plt.plot(first_vector_2.cpu().numpy(), label="SimRes - E/I (Sum)", color='green')
    plt.title("Comparison 1: Rebuild States vs SimRes")
    plt.xlabel("Index (Time Step)")
    plt.ylabel("Value")
    plt.legend()

    # 第二个对照（第二对曲线）
    plt.subplot(2, 1, 2)
    plt.plot(second_vector_1.cpu().numpy(), label="obs States - E/I (1)", color='red')
    plt.plot(second_vector_2.cpu().numpy(), label="Daily New Total EI", color='orange')
    plt.title("Comparison 2: Rebuild States vs Daily New Total EI")
    plt.xlabel("Index (Time Step)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _get_history_padding(datas, day, window_size):
    """
    获取填充后的历史窗口数据。

    当天数（day）大于或等于窗口大小（window_size）时，截取最近的 `window_size` 天的数据。
    当天数小于窗口大小时，在时间维度前端填充零，以确保返回的窗口大小为 `window_size`。

    参数:
        datas (torch.Tensor): 输入的观测数据，形状为 (env_count, total_days, ZONE_NUM)。
        day (int): 当前的天数（时间步），从1开始。
        window_size (int): 窗口大小，表示需要回溯的天数。

    返回:
        torch.Tensor: 填充后的窗口数据，形状为 (env_count, window_size, ZONE_NUM)。

    注意：
        返回的数据不包含索引为 day 的数据
    """
    if day >= window_size:
        # 获取最近的 `window_size` 天的数据
        window = datas[:, (day - window_size):day, :]  # 假设第3个特征是需要的观测值
    else:
        # 计算需要填充的天数
        padding = window_size - day
        # 获取已有的天数的数据
        current_data = datas[:, :day, :]  # 形状: (env_count, day, ZONE_NUM)
        # 在时间维度前端填充零
        pad = [0] * 2 * len(current_data.shape)
        pad[-4] = padding
        pad = tuple(pad)
        window = torch.nn.functional.pad(current_data, pad, "constant", 0)  # 形状: (env_count, window_size, ZONE_NUM)

    return window


def _load_dataset(dataset_dir):
    # 读取该文件夹下的所有文件
    # 获取所有以 'dataset' 开头且以 '.pth' 结尾的文件路径
    dataset_files = glob.glob(os.path.join(dataset_dir, 'dataset*.pth'))
    if not dataset_files:
        raise FileNotFoundError(f"在目录 '{dataset_dir}' 中未找到任何数据集文件。")

    # 初始化各个数据列表
    imperfect_obs_list = []
    history_action_list = []
    true_state_list = []

    # 遍历所有数据集文件并加载数据
    for file in dataset_files:
        data = torch.load(file)
        imperfect_obs_list.append(data['imperfect_obs'])
        history_action_list.append(data['history_action'])
        true_state_list.append(data['true_state'])


    # 合并各个数据列表
    imperfect_obs = torch.cat(imperfect_obs_list, dim=0)  # 形状: (总样本数, period + 1, ZONE_NUM)
    history_action = torch.cat(history_action_list, dim=0)  # 形状: (总样本数, period + 1, ZONE_NUM)
    true_state = torch.cat(true_state_list, dim=0)  # 形状: (总样本数, period + 1, ZONE_NUM)


    # 构建最终的数据集字典
    dataset = {
        'imperfect_obs': imperfect_obs,
        'history_action': history_action,
        'true_state': true_state,
    }


    # 打印数据集信息
    print("数据集已加载：")
    print(f"  imperfect_obs: {imperfect_obs.shape}")
    print(f"  history_action: {history_action.shape}")
    print(f"  true_state: {true_state.shape}")

    return dataset



def test(trainer, args, agent, state_norm):
    env = EpidemicModel(args, env_count=10)
    env.seed()
    device = env.device

    rebuild_states = torch.zeros((env.env_count, env.period + 1, env.ZONE_NUM, 2), device=device)

    s = env.reset()
    done = False
    while not done:
        history_action = _get_history_padding(env.actions, env.day - 1, 7)
        obs = _get_history_padding(env.history_local_obs[:, :, :, :2], env.day, 7) * env.POP.unsqueeze(
            1).unsqueeze(-1)
        # 重建信息
        pre = trainer.predict(obs, history_action)
        pre = torch.clip(pre, min=0)
        rebuild_states[:, env.day - 1, :, :] = pre
        # rebuild_states[:, env.day - 1, :, :] = env.history_local_obs[:, env.day - 1, :, :2] * env.POP.unsqueeze(-1)
        r_s = _get_history_padding(rebuild_states / env.POP.unsqueeze(1).unsqueeze(-1), env.day, env.WINDOW_SIZE)
        r_s = torch.cat((r_s, s[:, :, :, 2:]), dim=-1)
        s = r_s

        if args.use_state_norm:
            s = state_norm(s, update=False)
        a = agent.evaluate(s)
        # a, _ = agent.choose_action(s)  # 抽样动作
        # a = torch.randint(0, env.action_dim, (env.env_count, env.ZONE_NUM), device=env.device)
        # a = args.expert_policy.choose_action(s)

        s_, r, done, _ = env.step(a)

        s = s_

    if args.show_fig:
        env.render()
        from uncertainty.obs_imperfect.evaluate_info_rebuild import plot_all_env
        plot_all_env(rebuild_states, env)
        # for i in range(20):
        #     plot_one_region(rebuild_states, env, env_idx=0, region_idx=i)
        # print("对照：", rebuild_states[0, :, :, 0].sum(dim=1), env.simRes[0, :, :, [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(dim=[-2,-1]))
        # print("对照：", rebuild_states[0, :, :, 1].sum(dim=1), env.daily_new_total_EI[0, :, :].sum(dim=-1))
        #
        # # 对照1
        # first_vector_1 = rebuild_states[:, :, :, 0].sum(dim=2).mean(dim=0)  # shape (120,)
        # first_vector_2 = env.simRes[:, :, :, [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(dim=[-2,-1]).mean(dim=0)
        # first_vector_3 = env.simRes[:, :, :, [env.E_detected, env.I_detected, env.I_reported]].sum(dim=[-2,-1]).mean(dim=0)
        #
        # # 对照2
        # second_vector_1 = rebuild_states[:, :, :, 1].sum(dim=2).mean(dim=0)  # shape (120,)
        # second_vector_2 = env.daily_new_total_EI[:, :, :].sum(dim=-1).mean(dim=0)  # shape (120,)
        # second_vector_3 = env.daily_estimate_new_EI[:, :, :].sum(dim=-1).mean(dim=0)
        #
        # # 绘制图形
        # plt.figure(figsize=(10, 8))
        #
        # # 第一个对照（第一对曲线）
        # plt.subplot(2, 1, 1)
        # plt.plot(first_vector_1.cpu().numpy(), label="Rebuild States - E/I (0)", color='blue')
        # plt.plot(first_vector_2.cpu().numpy(), label="SimRes - E/I (Sum)", color='green')
        # plt.plot(first_vector_3.cpu().numpy(), label="SimRes - E/I (obs)", color='orange')
        # plt.title("Comparison 1: Rebuild States vs SimRes")
        # plt.xlabel("Index (Time Step)")
        # plt.ylabel("Value")
        # plt.legend()
        #
        # # 第二个对照（第二对曲线）
        # plt.subplot(2, 1, 2)
        # plt.plot(second_vector_1.cpu().numpy(), label="Rebuild States - E/I (1)", color='red')
        # plt.plot(second_vector_2.cpu().numpy(), label="Daily New Total EI", color='orange')
        # plt.plot(second_vector_3.cpu().numpy(), label="Daily Estimate New EI (obs)", color='purple')
        # # plt.plot(second_vector_4.cpu().numpy(), label="Daily Detect EI", color='blue')
        # plt.title("Comparison 2: Rebuild States vs Daily New Total EI")
        # plt.xlabel("Index (Time Step)")
        # plt.ylabel("Value")
        # plt.legend()
        #
        # # 显示图形
        # plt.tight_layout()
        # plt.show()

    # # TODO: 绘图 不完全观测：env.imperfect_obs[:, :, :, 2]  true_state: env.simRes[:, :, :, 2]  rebuild_states: rebuild_states
    # imperfect_obs = env.imperfect_states[:, :, :, 2]
    # true_state = env.simRes[:, :, :, 2]
    # rebuild_states = rebuild_states * env.state_standard_scale # 还原为真实人数
    #
    # # 创建时间轴
    # time_steps = range(env.period + 1)
    #
    # # 1. 绘制所有环境、所有区域的平均
    # plot_all_env(time_steps, imperfect_obs, true_state, rebuild_states)
    #
    # # 2. 绘制某个环境的平均值（跨所有区域）
    # plot_one_env(time_steps, imperfect_obs, true_state, rebuild_states, env_index=0)
    #
    # # 3. 绘制某个环境、某个区域的值
    # for i in range(env.ZONE_NUM):
    #     plot_one_region(time_steps, imperfect_obs, true_state, rebuild_states, env_index=0, zone_index=i)



def main_gru_gnn(args):
    from uncertainty.obs_imperfect.gru_gnn_model import RebuildGruGNN, RebuildGruGNNModel
    # args.device_name = 'cpu'

    # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
    env_temp = EpidemicModel(args, env_count=1)

    # 2.加载智能体
    args.use_obs_imperfect = False  # 要加载的是确定环境下的智能体
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    print("load directory:", agent.directory)
    agent.load(18)
    dataset_dir = os.path.join(agent.directory, 'dataset_gnn')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # 3.状态归一化state_norm加载
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 4.定义数据集收集
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    # # 数据集收集：1.智能体的动作产生的数据
    # for model_idx in range(1, 101):
    #     agent.load(model_idx)
    #     print(f'model_idx: {model_idx}')
    #     dataset = collect_dataset(args, repeat_times=1, env_count=2, agent=agent, state_norm=state_norm)
    #     # 保存到agent.directory下的dataset目录下,以dataset_{model_idx}命名
    #     torch.save(dataset, dataset_dir + f'/dataset{model_idx}.pth')
    # dataset = _load_dataset(dataset_dir)
    # agent.load(18)
    # dataset = collect_dataset(args, repeat_times=1, env_count=20, agent=agent, state_norm=state_norm)

    # 数据集收集：2.随机动作的数据集
    dataset = collect_dataset(args, repeat_times=1, env_count=20)

    # 获取数据集的一些维度信息
    num_samples, total_days, zone_num, _ = dataset['imperfect_obs'].shape
    od_zone_num = env_temp.ZONE_NUM  # OD 流动数据的区域数量
    assert zone_num == od_zone_num, "区域数量不一致"

    # 5.定义预测模型
    # 5.1.计算输入和输出的维度
    seq_len = 7
    hidden_size = 128
    output_size = 1
    num_layers = 2

    # 5.2.初始化模型
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers, env=env_temp, device_name=args.device_name)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    # 5.3.训练模型
    trainer.train(dataset, num_epochs=100, batch_size=64, patience=5)

    # 保存模型
    trainer.save('rebuild_gru_gnn_model.pth')

    trainer.load('rebuild_gru_gnn_model.pth')

    # 6.绘图测试重建效果
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1.0
    args.show_fig = True
    test(trainer, args, agent, state_norm=state_norm)


def main_gru_gnn_v1(args):
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNN, RebuildGruGNNModel
    # args.device_name = 'cpu'

    # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
    env_temp = EpidemicModel(args, env_count=1)

    # 2.加载智能体
    args.use_obs_imperfect = False  # 要加载的是确定环境下的智能体
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    print("load directory:", agent.directory)
    agent.load(args.model_idx)
    dataset_dir = os.path.join(agent.directory, 'dataset_gnn')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # 3.状态归一化state_norm加载
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 4.定义数据集收集
    args.use_obs_imperfect = False
    dataset1 = collect_dataset(args, repeat_times=1, env_count=20, agent=agent, state_norm=state_norm)

    # 数据集收集：2.随机动作的数据集
    dataset2 = collect_dataset(args, repeat_times=1, env_count=20)

    dataset = dataset1

    # 获取数据集的一些维度信息
    num_samples, total_days, zone_num, _ = dataset['imperfect_obs'].shape
    od_zone_num = env_temp.ZONE_NUM  # OD 流动数据的区域数量
    assert zone_num == od_zone_num, "区域数量不一致"

    # 定义方法，训练三种类型的gnn：
    def do_train(agent, args, dataset, env_temp, hidden_size, num_layers, output_size, seq_len, state_norm, model_name):
        # 5.2.初始化模型
        model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                                   env=env_temp, device_name=args.device_name)
        trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
        # 5.3.训练模型
        trainer.train(dataset, num_epochs=100, batch_size=64, patience=5)
        # 保存模型
        model_path = os.path.join(agent.directory, model_name)
        trainer.save(model_path)
        trainer.load(model_path)
        # 6.绘图测试重建效果
        args.use_obs_imperfect = True
        args.obs_imperfect_down = 0.1
        args.obs_imperfect_up = 1.0
        args.show_fig = True
        test(trainer, args, agent, state_norm=state_norm)

    # 5.定义预测模型
    # 5.1.计算输入和输出的维度
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2

    # # agent 动作收集的数据集
    do_train(agent, args, dataset1, env_temp, hidden_size, num_layers, output_size, seq_len, state_norm, 'rebuild_gru_gnn_model_v1_agent.pth')
    # # 随机动作收集的数据集
    do_train(agent, args, dataset2, env_temp, hidden_size, num_layers, output_size, seq_len, state_norm, 'rebuild_gru_gnn_model_v1_random.pth')
    # 普通的GNN
    env_temp.POP = None
    do_train(agent, args, dataset2, env_temp, hidden_size, num_layers, output_size, seq_len, state_norm, 'rebuild_gru_gnn_model_v1_ordinary.pth')





def main_gru_gnn_with_action(args):
    from uncertainty.obs_imperfect.gru_gnn_model_v2 import RebuildGruGNN, RebuildGruGNNModel
    # args.device_name = 'cpu'

    # 初始化一个环境，使用其 OD POP数据，以及action_to_u方法
    env_temp = EpidemicModel(args, env_count=1)

    # 2.加载智能体
    args.max_train_steps = 2400 * 20
    args.use_obs_imperfect = False  # 要加载的是确定环境下的智能体
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    print("load directory:", agent.directory)
    agent.load(args.model_idx)
    dataset_dir = os.path.join(agent.directory, 'dataset_gnn')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # 3.状态归一化state_norm加载
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 4.定义数据集收集
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    # 数据集收集：1.智能体的动作产生的数据
    # for model_idx in range(1, 101):
    #     agent.load(model_idx)
    #     print(f'model_idx: {model_idx}')
    #     dataset = collect_dataset(args, repeat_times=1, env_count=2, agent=agent, state_norm=state_norm)
    #     # 保存到agent.directory下的dataset目录下,以dataset_{model_idx}命名
    #     torch.save(dataset, dataset_dir + f'/dataset{model_idx}.pth')
    # dataset = _load_dataset(dataset_dir)
    agent.load(args.model_idx)
    args.use_obs_imperfect = False
    dataset1 = collect_dataset(args, repeat_times=1, env_count=20, agent=agent, state_norm=state_norm)

    # 数据集收集：2.随机动作的数据集
    dataset2 = collect_dataset(args, repeat_times=1, env_count=20)

    # # 拼接两种数据集
    # dataset = {}
    # for key in dataset1.keys():
    #     dataset[key] = torch.cat([dataset1[key], dataset2[key]], dim=0)

    dataset = dataset1

    # 获取数据集的一些维度信息
    num_samples, total_days, zone_num, _ = dataset['imperfect_obs'].shape
    od_zone_num = env_temp.ZONE_NUM  # OD 流动数据的区域数量
    assert zone_num == od_zone_num, "区域数量不一致"

    # 5.定义预测模型
    # 5.1.计算输入和输出的维度
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2

    # 5.2.初始化模型
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name, node_output_size=48)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len, env=env_temp)
    # 5.3.训练模型
    trainer.train(dataset, num_epochs=100, batch_size=64, patience=5)

    # 保存模型
    model_path = os.path.join(agent.directory, 'rebuild_gru_gnn_model_with_action.pth')
    trainer.save(model_path)

    trainer.load(model_path)

    # 6.绘图测试重建效果
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1.0
    args.show_fig = True
    test(trainer, args, agent, state_norm=state_norm)


if __name__ == '__main__':
    from config import args
    _config_args()
    args.env_data_dir = '../../data/'
    args.daily_imported_cases = 0
    args.model_idx = 18

    args.collect_action = True

    # # 低R0实验
    # args.ODE_beta = 0.4
    # args.R0 = 'low'

    # main_gru_gnn(args)
    main_gru_gnn_v1(args)
    # main_gru_gnn_with_action(args)