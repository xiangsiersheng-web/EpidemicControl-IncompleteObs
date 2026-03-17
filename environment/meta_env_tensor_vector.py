# -*- encoding: utf-8 -*-
# @Time    : 2024-08-31
# @File    : meta_env_tensor_vector.py

import random
import numpy as np
import matplotlib.pyplot as plt
import torch


config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 16,
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)


class EpidemicModelTensorVector:
    def __init__(self, args, env_count=20):
        args_dict = vars(args)
        self.reward_mode = args.reward_mode
        self.city = args.city
        self.R0 = args.R0
        env_count = int(env_count)
        self.env_count = env_count
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device_name)

        data_dir = '../data/' if 'env_data_dir' not in args_dict else args.env_data_dir
        data_dir += self.city
        # 子区域邻接关系
        self.adj_matrix = torch.from_numpy(np.load(data_dir + '/adj_matrix.npy')).float().to(self.device)
        self.flow_top3_matrix = torch.from_numpy(np.load(data_dir + '/flow_top3_matrix.npy')).float().to(self.device)
        if self.city == 'sz':
            self.adm_matrix = torch.from_numpy(np.load(data_dir + '/adm_matrix.npy')).float().to(self.device)
        self.action_max = 3  # 注意，action的最大值是写死了的

        # 子区域间流动
        self.OD = torch.from_numpy(np.load(data_dir + '/flow.npy')).float().to(self.device)

        # 子区域人口数量
        self.POP = torch.from_numpy(np.load(data_dir + '/population.npy')).float().to(self.device)
        self.ZONE_NUM = self.POP.shape[0]

        # 给矩阵增加一个维度，用于后续的计算操作
        self.adj_matrix = self.adj_matrix.unsqueeze(0).expand(env_count, -1, -1)
        self.flow_top3_matrix = self.flow_top3_matrix.unsqueeze(0).expand(env_count, -1, -1)
        if self.city == 'sz':
            self.adm_matrix = self.adm_matrix.unsqueeze(0).expand(env_count, -1, -1)
        self.OD = self.OD.unsqueeze(0).expand(env_count, -1, -1)
        self.contagious_OD = self.OD
        self.POP = self.POP.unsqueeze(0).expand(env_count, -1)

        # 传染病相关参数（感染者移动比例、beta、潜伏期、恢复期）
        self.Pm = torch.tensor(0.4 if "ODE_Pm" not in args_dict else args.ODE_Pm, device=self.device)
        self.beta = 0.8 if "ODE_beta" not in args_dict else args.ODE_beta
        self.betas = torch.full((self.ZONE_NUM, ), self.beta, device=self.device)
        self.betas = self.betas.unsqueeze(0).expand(env_count, -1)
        self.sigma = torch.tensor(1 / 3 if "ODE_sigma" not in args_dict else args.ODE_sigma, device=self.device)
        self.gamma = torch.tensor(1 / 7 if "ODE_gamma" not in args_dict else args.ODE_gamma, device=self.device)

        self.use_beta_change = False
        self.beta_matrix = None
        self.state_contain_beta = False if "state_contain_beta" not in args_dict else args.state_contain_beta
        self.state_is_sequence = False # 是否使用序列状态
        if args.actor_critic_model == "lstm":
            self.state_is_sequence = True

        # 不同城市的容量和ylim（绘图用）
        city_capacity = {'sz': 4e6, 'tokyo': 2.2e6, 'nyc': 2e6, 'sh': 5.5e6}
        self.capacity = torch.tensor(city_capacity[self.city], device=self.device)
        self.capacity = self.capacity if self.R0 == 'high' else self.capacity / 2

        city_ylim = {'sz': 9e6, 'tokyo': 6e6, 'nyc': 6e6, 'sh': 1.2e7}
        self.ylim = city_ylim[self.city]

        # 模拟周期
        self.period = 120 if "ODE_period" not in args_dict else args.ODE_period

        # 状态的观测天数
        self.WINDOW_SIZE = 7 if "WINDOW_SIZE" not in args_dict else args.WINDOW_SIZE

        # 是否对I进行不完全观测
        self.I_obs_imperfect = False if "I_obs_imperfect" not in args_dict else args.I_obs_imperfect
        self.I_obs_imperfect_up = 1.0 if "I_obs_imperfect_up" not in args_dict else args.I_obs_imperfect_up
        self.I_obs_imperfect_down = 1.0 if "I_obs_imperfect_down" not in args_dict else args.I_obs_imperfect_down
        # 是否基于不完全观测计算奖励
        self.use_imperfect_calc_reward = False if "use_imperfect_calc_reward" not in args_dict else args.use_imperfect_calc_reward

        # 异常时空行为
        self.gather_to_some_region = False if "gather_to_some_region" not in args_dict else args.gather_to_some_region # I向某些区域聚集


    def adjust_beta_matrix(self, beta_matrix):
        """
        调整beta矩阵，用于动态调整beta矩阵。
        :param beta_matrix: tensor, shape: (env_count, period, zone_num)
        """
        # beta矩阵应该为tensor，且在device上
        assert isinstance(beta_matrix, torch.Tensor) and beta_matrix.device == self.device, "beta_matrix should be a tensor and on the same device as the environment."
        print("使用了beta变化")
        self.use_beta_change = True
        self.beta_matrix = beta_matrix

    def dynamic_beta_change(self, type, std):
        if type not in ['none', 'spatial', 'temporal', 'spatiotemporal']:
            raise ValueError('Invalid beta change type: ', type)
        if type == 'none':
            self.use_beta_change = False
            return
        else:
            self.use_beta_change = True

        # 根据type类型和std值，为每个环境生成随机beta矩阵
        if type == 'spatial':
            beta_spatial = torch.full((self.ZONE_NUM, ), self.beta, device=self.device)
            self.beta_matrix = torch.zeros((self.env_count, self.period, self.ZONE_NUM)).to(self.device)
            # 直接为所有批次生成噪声
            noise = torch.rand(self.env_count, self.ZONE_NUM, device=self.device) * std
            beta_spatial_noise = beta_spatial + noise
            self.beta_matrix += beta_spatial_noise.unsqueeze(1).expand(-1, self.period, -1)
        elif type == 'temporal':
            beta_temporal = torch.full((self.period, ), self.beta, device=self.device)
            self.beta_matrix = torch.zeros((self.env_count, self.period, self.ZONE_NUM)).to(self.device)
            noise = torch.rand(self.env_count, self.period, device=self.device) * std
            beta_temporal_noise = beta_temporal + noise # shape: (env_count, period)
            self.beta_matrix += beta_temporal_noise.unsqueeze(2).expand(-1, -1, self.ZONE_NUM)
        elif type == 'spatiotemporal':
            raise NotImplementedError
        pass

    def adjust_env_params(self, **kwargs):
        """用于动态调用环境参数（废弃）"""
        raise NotImplementedError
        # if 'beta' in kwargs and 'betas' in kwargs:
        #     raise ValueError("Cannot specify both 'beta' and 'betas'.")
        # if 'beta' in kwargs:
        #     if (self.beta != kwargs['beta']):
        #         self.beta = kwargs['beta']
        #         self.betas = np.array([self.beta] * self.ZONE_NUM)
        # if 'betas' in kwargs:
        #     self.betas = np.array(kwargs['betas'])
        # if len(self.betas) != self.ZONE_NUM:
        #     raise ValueError("Length of 'betas' must be equal to ZONE_NUM.")
        # # for k, v in kwargs.items():
        # #     setattr(self, k, v) # 调整对象的属性


    def reset(self, rand_idxs=None):
        """
        重置仿真环境到初始状态。

        此函数用于：
        - 将当前天数重置为1。
        - 初始化动作数组为区域数量的零向量。
        - 清空历史成本数据，包括reward, sdo, fdo, ado。
        - 设置仿真状态，其中包括人口数据初始化。
        - 重置仿真结果数组。
        - 计算初始观察状态，并返回。

        返回：
        - 返回观察数组，代表当前环境状态。
        """
        self.day = 1
        # 初始化动作数组，现在是三维数组：(env_count, period+1, ZONE_NUM)
        self.actions = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 初始化历史成本数据，每个成本数据维度为 (env_count, period)
        self.history_cost = {
            "reward": torch.zeros((self.env_count, self.period), device=self.device),
            "sdo": torch.zeros((self.env_count, self.period), device=self.device),
            "fdo": torch.zeros((self.env_count, self.period), device=self.device),
            "ado": torch.zeros((self.env_count, self.period), device=self.device)
        }

        # 初始化仿真状态，扩展为 (env_count, ZONE_NUM, 4)
        self.simState = torch.zeros((self.env_count, self.ZONE_NUM, 4), device=self.device).float()
        self.simState[:, :, 0] = self.POP

        # 设置初始感染者种子
        self.set_init_seed(rand_idxs=rand_idxs)

        # 初始化仿真结果数组，现在为四维数组：(env_count, period+1, ZONE_NUM, 4)
        self.simRes = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 4), device=self.device)
        self.simRes[:, 0, :, :] = self.simState

        # 随机化各个环境各个区域的检测率
        if self.I_obs_imperfect:
            self.construct_I_detect_rate()

        # 初始化不完全观测, 第四个维度上暂时只需要保存 I
        self.imperfect_obs = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 4), device=self.device)
        self.imperfect_obs[:, 0, :, 2] = self.imperfect_I()

        # 构建初始观察窗口
        padding = self.WINDOW_SIZE - self.day
        # 获取所有已经模拟的天数的数据
        current_data = self.imperfect_obs[:, :self.day, :, 2]  # (env_count, current day, ZONE_NUM)
        Is_window = torch.nn.functional.pad(current_data, (0, 0, padding, 0), "constant",
                                            0)  # (env_count, WINDOW_SIZE, ZONE_NUM)
        window = Is_window
        if self.state_contain_beta: # 如果包含beta，则将beta矩阵也加入窗口
            beta_window = self.beta_matrix[:, :(self.day-1), :]
            beta_window_padding = torch.nn.functional.pad(beta_window, (0, 0, padding+1, 0), "constant", 0)
            window = torch.cat([beta_window_padding, window], dim=2)  # (env_count, WINDOW_SIZE, 2 * ZONE_NUM)

        if self.state_is_sequence: # 如果要求状态是时间序列的，则不展平
            obs = window
        else:
            obs = window.view(self.env_count, -1)  # 展平每个环境的观察窗口

        obs = obs / 1e4

        # 初始化日常新发事件数组
        self.daily_new_E = torch.zeros((self.env_count, self.period), device=self.device)
        self.daily_new_I = torch.zeros((self.env_count, self.period), device=self.device)
        self.daily_attack_rate = torch.zeros((self.env_count, self.period), device=self.device)

        return obs

    def construct_I_detect_rate(self):
        # 各个区域的检测率 在 down 和 up 之间，有 浮动
        up = self.I_obs_imperfect_up
        down = self.I_obs_imperfect_down
        self.I_detect_rate = torch.rand((self.env_count, self.ZONE_NUM), device=self.device) * (up - down) + down

    def imperfect_I(self, delay = 0, std = 0.05):
        """
        计算不完全观察的I值。

        此函数用于：
        - 计算不完全观察的I值，通过在真实I值上乘以一个高斯分布的检测率。

        返回：
        - 返回不完全观察的I值。

        根据3σ原则：
        - 60%的概率在1σ内，95%的概率在2σ内，99%的概率在3σ内。
        """
        if not self.I_obs_imperfect:
            return self.simState[:, :, 2]

        # 1.延迟观测
        if self.day <= delay:
            imperfect_I = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)
        else:
            imperfect_I = self.simRes[:, self.day - 1 - delay, :, 2]

        # 模拟一些异常的观测
        # if self.day in range(30, 50):
        #     imperfect_I = self.simRes[:, 29, :, 2] * (1 - self.day / 50)

        # 2.对当天的观测不准
        # 为每个环境的每个区域每天的检测率添加随机高斯噪声
        daily_rate = self.I_detect_rate * (
                1 + torch.randn((self.env_count, self.ZONE_NUM), device=self.device) * std)
        # 计算不完全的观测I值
        imperfect_I = imperfect_I * daily_rate

        return imperfect_I

    def set_init_seed(self, init_infection=100, rand_idxs=None):
        """
        设置初始感染者种子。

        此函数用于：
        - 使用PyTorch的随机种子设定，以确保GPU兼容性。
        - 随机选择初始感染者位置。
        - 将初始感染者数量添加到初始状态中。
        """
        if rand_idxs is None:
            random.seed(3074)
            rand_list = [random.randint(0, self.ZONE_NUM - 1) for _ in range(init_infection)]
            rand_idxs = torch.tensor(rand_list, device=self.device)
            # 将单个环境的索引复制到每个批次
            rand_idxs = rand_idxs.repeat(self.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组


        # 初始化一个batch的计数器
        counts = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)

        # 使用scatter_add_进行并行累加，更新每个环境的感染者数量
        counts = counts.scatter_add_(1, rand_idxs, torch.ones_like(rand_idxs, dtype=torch.float))

        # 更新self.simState中对应索引的第一、二列
        self.simState[:, :, 1] += counts
        self.simState[:, :, 0] -= counts

    def step(self, action = None):
        """
        根据动作执行一步。

        此函数用于：
        - 天数+1
        - 根据动作更新环境状态。
        - 计算奖励。
        - 记录历史信息，在达到self.period天时，返回这条episode的历史信息

        返回：
        - 返回新的状态，奖励值，是否终止，episode的历史信息。
        """
        if action is None:
            action = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)
        # action 保证是tensor，且在device上
        assert type(action) == torch.Tensor and action.device == self.device, "action must be a tensor on device"
        # 确保action是(env_count, ZONE_NUM)的Tensor
        assert action.shape == (self.env_count, self.ZONE_NUM), "Action shape must be (env_count, ZONE_NUM)"
        if self.use_beta_change:
            # self.betas = self.beta_matrix[self.day - 1].unsqueeze(0).expand(self.env_count, -1)
            self.betas = self.beta_matrix[:, self.day - 1, :]
        self.day += 1
        self.actions[:, self.day - 1, :] = action # shape (env_count, ZONE_NUM)

        # 仓室模型step
        contagious_infects = self.simState[:, :, 1] + self.Pm * self.simState[:, :, 2]  # (env_count, ZONE_NUM)

        # 计算传染病传播到每个区域的影响，使用contagious_OD的转置
        contagious_toJ = torch.bmm(self.contagious_OD.transpose(2, 1),
                                   contagious_infects.unsqueeze(-1)).squeeze(-1) # (env_count, ZONE_NUM)
        contagious_toJ = torch.clamp(contagious_toJ, min=0.0)


        all_toJ = torch.bmm(self.OD.transpose(2, 1), self.POP.unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)
        contagious_ratio_toJ = contagious_toJ / all_toJ # (env_count, ZONE_NUM)
        contagious_ratio_toJ = torch.clamp(contagious_ratio_toJ, min=0.0)

        # 计算带有 betas 的 contagious_ratioToJ
        modified_ratio = contagious_ratio_toJ * self.betas * (1 - 0.25 * action) # (env_count, ZONE_NUM)
        lam = torch.bmm(self.OD, modified_ratio.unsqueeze(-1)).squeeze(-1) # (env_count, ZONE_NUM)
        lam = torch.clamp(lam, min=0.0)

        dS = -self.simState[:, :, 0] * lam
        dE = self.simState[:, :, 0] * lam - self.sigma * self.simState[:, :, 1]
        dI = self.sigma * self.simState[:, :, 1] - self.gamma * self.simState[:, :, 2]
        dR = self.gamma * self.simState[:, :, 2]

        # 更新状态
        dState = torch.stack([dS, dE, dI, dR], dim=2) # (env_count, ZONE_NUM, 4)

        # 计算新增事件
        addE = self.simState[:, :, 0] * lam # (env_count, ZONE_NUM)
        self.daily_new_E[:, self.day - 2] = addE.sum(dim=1) # (env_count,)
        addI = self.sigma * self.simState[:, :, 1]
        self.daily_new_I[:, self.day - 2] = addI.sum(dim=1)
        # 计算新增后再更新状态
        self.simState += dState
        self.simRes[:, self.day - 1, :, :] = self.simState
        self.imperfect_obs[:, self.day - 1, :, 2] = self.imperfect_I()

        # 构造state/obs
        if self.day >= self.WINDOW_SIZE:
            Is_window = self.imperfect_obs[:, (self.day - self.WINDOW_SIZE):self.day, :, 2]  # (env_count, WINDOW_SIZE, ZONE_NUM)
            window = Is_window
            if self.state_contain_beta:
                if self.day == self.WINDOW_SIZE: # 此时还需要填充
                    beta_window = self.beta_matrix[:, :(self.day-1), :]
                    beta_window = torch.nn.functional.pad(beta_window, (0, 0, 1, 0), "constant", 0)
                else:
                    beta_window = self.beta_matrix[:, (self.day - self.WINDOW_SIZE - 1):(self.day - 1), :]
                window = torch.cat([beta_window, window], dim=2) # (env_count, WINDOW_SIZE, 2 * ZONE_NUM)
        else:
            padding = self.WINDOW_SIZE - self.day
            # 获取所有已经模拟的天数的数据
            current_data = self.imperfect_obs[:, :self.day, :, 2]  # (env_count, current day, ZONE_NUM)
            Is_window = torch.nn.functional.pad(current_data, (0, 0, padding, 0), "constant", 0)  # (env_count, WINDOW_SIZE, ZONE_NUM)
            window = Is_window
            if self.state_contain_beta:
                beta_window = self.beta_matrix[:, :(self.day-1), :]
                beta_window_padding = torch.nn.functional.pad(beta_window, (0, 0, padding+1, 0), "constant", 0)
                window = torch.cat([beta_window_padding, window], dim=2) # (env_count, WINDOW_SIZE, 2 * ZONE_NUM)

        if self.state_is_sequence:  # 如果要求状态是时间序列的，则不展平
            obs = window
        else:
            obs = window.view(self.env_count, -1)  # 展平每个环境的观察窗口

        obs = obs / 1e4

        # 计算奖励
        reward = self.reward_func(action)

        info = {}
        done = False
        if self.day > self.period:
            # 计算指标
            total_infections = self.simRes[:, :, :, 2].sum(dim=2)  # (env_count, period+1)
            ep_r = self.history_cost['reward'].sum(dim=1)  # (env_count,)
            ep_overload = (
                        (total_infections.max(dim=1).values - self.capacity) / self.capacity).squeeze()  # (env_count,)
            ep_intensity = self.actions.sum(dim=[1, 2])  # (env_count,)
            ep_sdo = self.history_cost['sdo'].sum(dim=1)  # (env_count,)
            ep_fdo = self.history_cost['fdo'].sum(dim=1)  # (env_count,)
            ep_tdo = torch.sum(torch.abs(self.actions[:, 1:] - self.actions[:, :-1]), dim=2).sum(dim=1)  # (env_count,)
            ep_ado = self.history_cost['ado'].sum(dim=1)  # (env_count,)

            # 构造结果字典
            # info = {
            #     'ep_r': ep_r.tolist(),
            #     'ep_overload': ep_overload.tolist(),
            #     'ep_intensity': ep_intensity.tolist(),
            #     'ep_sdo': ep_sdo.tolist(),
            #     'ep_fdo': ep_fdo.tolist(),
            #     'ep_tdo': ep_tdo.tolist(),
            #     'ep_ado': ep_ado.tolist()
            # }
            info = {
                'ep_r': ep_r,
                'ep_overload': ep_overload,
                'ep_intensity': ep_intensity,
                'ep_sdo': ep_sdo,
                'ep_fdo': ep_fdo,
                'ep_tdo': ep_tdo,
                'ep_ado': ep_ado
            }

            done = True

        # 理论上done也要批次化，但对于这个环境，大家都是一起开始，一起结束的，所以done只是一个标量
        return obs, reward, done, info

    def adjust_contagious_OD(self, regions = [0], increase_ratio = [0.5]):
        """
        此函数用于调整contagious_OD
        :param regions: 要调整的区域
        :param increase_ratio: 要提高的比例
        """
        # 确保regions和increase_ratio列表长度一致
        assert len(regions) == len(increase_ratio), "regions和increase_ratio列表长度需相等"

        self.contagious_OD = self.OD.clone()
        # 遍历每个需要调整的区域及其对应的增加比例
        for region, ratio in zip(regions, increase_ratio):
            # 增加指定区域的流入比例
            self.contagious_OD[:, :, region] *= (1 + ratio)

        # 重新归一化以保持每个区域的流出比例总和为1
        # 注意：这种计算较简洁，但并不能真正提高到预定的比例，会略低一点
        sum_along_dim2 = self.contagious_OD.sum(dim=2, keepdim=True)
        self.contagious_OD /= sum_along_dim2

        # 输出提高比例
        for region in regions:
            print(f"Region {region}: from {self.OD[:, :, region].sum(dim=1).mean(dim=0)} to {self.contagious_OD[:, :, region].sum(dim=1).mean(dim=0)}")


    def reset_contagious_OD(self):
        """
        重置contagious_OD为原始的OD矩阵
        :return:
        """
        self.contagious_OD = self.OD

    def reward_func(self, action):
        if self.use_imperfect_calc_reward:
            state = self.imperfect_obs[:, self.day - 1, :, :]
        else:
            state = self.simRes[:, self.day - 1, :, :]
        # 计算奖励
        total_infections = state[:, :, 2].sum(dim=1)  # (env_count,)
        health_cost = -torch.clamp((total_infections - self.capacity) / self.capacity, min=0)

        # 计算经济成本
        economy_cost = -action.sum(dim=1) / self.ZONE_NUM  # (env_count,)

        # 计算秩序成本
        t_order_cost = -torch.abs(action - self.actions[:, -2, :]).sum(dim=1) / self.ZONE_NUM  # (env_count,)
        s_order_cost = -self.spatio_entropy_adj_matrix(action,
                                                       self.adj_matrix) / self.ZONE_NUM  # 假设这个函数返回的是 (env_count,)
        f_order_cost = -self.spatio_entropy_adj_matrix(action, self.flow_top3_matrix) / self.ZONE_NUM  # 同上

        if self.city == 'sz':
            a_order_cost = -self.spatio_entropy_adm(action, self.adm_matrix) / self.ZONE_NUM  # 假设这个函数返回的是 (env_count,)
        else:
            a_order_cost = torch.zeros((self.env_count,), device=self.device)

        if self.reward_mode == 4:
            order_cost = torch.zeros((self.env_count,), device=self.device)  # 无秩序-4
        elif self.reward_mode == 2:
            order_cost = t_order_cost * 2  # t时间秩序-2
        elif self.reward_mode == 3:
            order_cost = s_order_cost * 2  # s空间秩序-3
        elif self.reward_mode == 5:
            order_cost = f_order_cost * 2  # f秩序-5
        elif self.reward_mode == -1:
            order_cost = a_order_cost * 2  # a秩序- -1
        elif self.reward_mode == 1:
            order_cost = t_order_cost + s_order_cost  # st秩序-1
        elif self.reward_mode == 6:
            order_cost = t_order_cost + f_order_cost  # ft秩序-6
        elif self.reward_mode == -2:
            order_cost = t_order_cost + a_order_cost  # at秩序- -2

        reward = health_cost * 60 + economy_cost + order_cost * 2

        # 记录奖励，确保为每个批次独立记录
        self.history_cost['reward'][:, self.day - 2] = reward
        self.history_cost['sdo'][:, self.day - 2] = -s_order_cost
        self.history_cost['fdo'][:, self.day - 2] = -f_order_cost
        self.history_cost['ado'][:, self.day - 2] = -a_order_cost

        return reward

    def spatio_entropy_adj_matrix(self, action, adj_matrix):
        """
        使用优化的Tensor操作计算空间熵。
        :param action: Tensor, 各节点的动作值（整数1到3）, 形状为 (env_count, ZONE_NUM)
        :param adj_matrix: Tensor, 邻接矩阵, 形状为 (env_count, ZONE_NUM, ZONE_NUM)
        :return: Tensor, 计算得到的空间熵，形状为 (env_count,)
        """
        epsilon = 1e-9  # 防止除零和log(0)

        # 扩展action到每行都是完整的action复制
        extended_action = action.unsqueeze(1).expand(-1, adj_matrix.size(1), -1) # (env_count, ZONE_NUM, ZONE_NUM)

        # 应用邻接矩阵，过滤邻接节点的行为
        filtered_actions = extended_action * adj_matrix # (env_count, ZONE_NUM, ZONE_NUM)

        # 将非邻接的部分设置为0
        filtered_actions[adj_matrix == 0] = -1  # 使用不在[0, action_max)范围内的数字标记非邻接

        # 计算每个动作的频率
        action_counts = torch.zeros(self.env_count, filtered_actions.size(1), self.action_max, device=filtered_actions.device)
        for i in range(0, self.action_max):
            action_counts[:, :, i] = (filtered_actions == i).sum(dim=2)

        # 计算概率
        row_sums = action_counts.sum(dim=2, keepdim=True)  # Sum across all action types for each batch
        probabilities = action_counts / row_sums.clamp(min=epsilon) # (env_count, ZONE_NUM, action_max)

        # 计算熵
        entropy = -(probabilities * torch.log2(probabilities + epsilon))
        row_entropy = entropy.sum(dim=2) # (env_count, ZONE_NUM)
        total_entropy = row_entropy.sum(dim=1) # (env_count,)

        return total_entropy

    def spatio_entropy_adm(self, action, adm_matrix):
        """
        使用优化的Tensor操作计算空间熵，专用于街道
        :param action: Tensor, 各节点的动作值（整数1到3）, 形状为 (env_count, zone_num)
        :param adm_matrix: Tensor, 街道邻接, 形状为 (env_count, direct_num, zone_num)
        :return: Tensor, 计算得到街道层面的空间熵, 形状为 (env_count,)
        """
        epsilon = 1e-9  # 防止除零和log(0)

        # 扩展action到每行都是完整的action复制
        extended_action = action.unsqueeze(1).expand(-1, adm_matrix.size(1), -1)  # (env_count, direct_num, zone_num)

        # 应用邻接矩阵，过滤邻接节点的行为
        filtered_actions = extended_action * adm_matrix  # (env_count, direct_num, zone_num)

        # 将非邻接的部分设置为-1，表示这不是一个有效的动作
        filtered_actions[adm_matrix == 0] = -1

        # 计算每个动作的频率
        action_counts = torch.zeros(self.env_count, filtered_actions.size(1), self.action_max, device=filtered_actions.device) # (env_count, direct_num, action_max)
        for i in range(0, self.action_max):
            action_counts[:, :, i] = (filtered_actions == i).sum(dim=2)

        # 计算概率
        row_sums = action_counts.sum(dim=2, keepdim=True) # (env_count, direct_num, 1)
        probabilities = action_counts / row_sums # (env_count, direct_num, action_max)

        # 计算熵
        entropy = -(probabilities * torch.log2(probabilities + epsilon))

        row_entropy = entropy.sum(dim=2) # (env_count, direct_num)
        # adm 矩阵的行和作为权重
        w = adm_matrix.sum(dim=2) # (env_count, direct_num)
        total_entropy = (row_entropy * w).sum(dim = 1) # (env_count,)

        return total_entropy

    def render(self, title='') :
        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        # 绘制不完美观测
        I_imperfect = self.imperfect_obs[:, :, :, 2].sum(dim=2).mean(dim=0)
        plt.plot(I_imperfect, label='Infections (imperfect)', color='red', linewidth=2)

        # 计算所有环境的感染总和并取平均
        infections = self.simRes[:, :, :, 2].sum(dim=2).mean(dim=0)  # (period+1,)
        plt.plot(infections, label='Infections (avg)', color='orange', linewidth=2)

        plt.axhline(self.capacity.item(), ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, self.ylim)

        plt.legend(fontsize=8, loc='upper left')
        plt.twinx()

        # 计算所有批次的NPI平均值
        actions_avg = self.actions.mean(dim=0).mean(dim=1)  # (period+1,)
        plt.plot(actions_avg, label='NPI (avg)', color='green', linewidth=2)

        # 绘制动作
        plt.plot(self.actions.mean(dim=0), alpha=0.3, color='0.5')

        plt.ylim(-1, 3)
        plt.yticks(range(0, 3, 1))

        plt.title("Daily Current Infection (avg) (" + self.city + ")" if title == '' else title)

        plt.legend(fontsize=8, loc='upper right')

        plt.show()

    def render_one_env(self, title='', env_idx=0) :
        plt.figure(dpi=120, figsize=(4.2, 3.6))
        plt.grid(linestyle='-.', axis='both')

        # 绘制不完美观测
        I_imperfect = self.imperfect_obs[env_idx, :, :, 2].sum(dim=1)
        plt.plot(I_imperfect, label='Infections (imperfect)', color='red', linewidth=2)

        # 计算所有环境的感染总和并取平均
        infections = self.simRes[env_idx, :, :, 2].sum(dim=1)  # (period+1,)
        plt.plot(infections, label='Infections', color='orange', linewidth=2)

        plt.axhline(self.capacity.item(), ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, self.ylim)

        plt.legend(fontsize=8, loc='upper left')
        plt.twinx()

        # 计算所有批次的NPI平均值
        actions_avg = self.actions[env_idx, :, :].mean(dim=1)  # (period+1,)
        plt.plot(actions_avg, label='NPI (avg)', color='green', linewidth=2)

        # 绘制动作
        plt.plot(self.actions[env_idx, :, :], alpha=0.3, color='0.5')

        plt.ylim(-1, 3)
        plt.yticks(range(0, 3, 1))

        plt.title(f"Daily Current Infection (env idx={env_idx}) (" + self.city + ")" if title == '' else title)

        plt.legend(fontsize=8, loc='upper right')

        plt.show()

    def render_one_region(self, title='', env_idx=0, region_idx=0) :
        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        # 绘制不完美观测
        I_imperfect = self.imperfect_obs[env_idx, :, region_idx, 2]
        plt.plot(I_imperfect, label='Infections (imperfect)', color='red', linewidth=2)

        # 该区域的感染曲线
        infections = self.simRes[env_idx, :, region_idx, 2]  # (period+1,)
        plt.plot(infections, label='Infections', color='orange', linewidth=2)

        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, 6.5e4)

        plt.legend(fontsize=14, loc='upper left')
        plt.twinx()

        # 计算NPI平均值
        actions_avg = self.actions[env_idx, :, region_idx]  # (period+1,)
        plt.plot(actions_avg, label='NPI', color='green', linewidth=2)

        # 绘制动作
        plt.plot(self.actions[env_idx, :, region_idx], alpha=0.3, color='0.5')

        # 输出动作变化的天数
        change_days = []
        for i in range(1, len(actions_avg) - 1):
            if actions_avg[i] != actions_avg[i - 1]:
                change_days.append(i)
        print("动作变化的天数",change_days)

        plt.ylim(-1, 3)
        plt.yticks(range(0, 3, 1))

        plt.title(f"Daily Current Infection (env idx={env_idx}, region idx={region_idx}) (" + self.city + ")" if title == '' else title)

        plt.legend(fontsize=14, loc='upper right')

        plt.show()

    def close(self):
        pass




if __name__ == '__main__':

    # # 1.测试耗时
    # from config import args
    #
    # args.device_name = 'cpu'
    # env = EpidemicModelGpuVector(args, env_count=1000)
    # actions = np.ones((env.env_count, 120, env.ZONE_NUM))
    # actions = torch.from_numpy(actions).float().to(env.device)
    #
    # random.seed(3074)
    # rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    # rand_idxs = torch.tensor(rand_list, device=env.device)
    # # 将单个环境的索引复制到每个批次
    # rand_idxs = rand_idxs.repeat(env.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组
    #
    # import time
    #
    # start_time = time.time()
    # for i in range(5):
    #     env.reset(rand_idxs)
    #     ep_s = 0
    #     ep_r = torch.zeros((env.env_count,), device=env.device)
    #     while True:
    #
    #         s_, r, done, info = env.step(action=actions[:, ep_s, :] * (i % 3))
    #         ep_s += 1
    #         ep_r += r
    #
    #         if done:
    #             ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_ado, ep_tdo = info.values()
    #
    #             # 总感染的峰值天数需要对每个批次进行计算
    #             peak_days = torch.argmax(env.simRes[:, :, :, 2].sum(dim=2), dim=1).tolist()  # (env_count,)
    #
    #             for batch_index in range(env.env_count):
    #                 print(
    #                     f"Batch {batch_index}, level {i} || reward: {ep_r[batch_index]:.4f}\t"
    #                     f"overload: {ep_overload[batch_index]:.4f}\t intensity: {ep_intensity[batch_index]:.2f}\t"
    #                     f"tdo: {ep_tdo[batch_index]:.4f}\t sdo: {ep_sdo[batch_index]:.4f}\t"
    #                     f"fdo: {ep_fdo[batch_index]:.2f}\t ado: {ep_ado[batch_index]:.2f}\t"
    #                     f"peak_day: {peak_days[batch_index]}"
    #                 )
    #
    #             break
    #
    # print("--- 耗时： %s seconds ---" % (time.time() - start_time))


    # 2.测试一致性 （应该是十分一致的）
    # from config import args
    # # args.I_obs_imperfect = True
    # # args.I_obs_imperfect_down = 0.5
    # args.device_name = 'cpu'
    # env = EpidemicModelTensorVector(args, env_count=10)
    # actions = np.ones((env.env_count, 120, env.ZONE_NUM)) * 2
    # np.random.seed(3047)
    # action1 = np.random.randint(0, env.action_max, env.ZONE_NUM)
    # np.random.seed(305)
    # action2 = np.random.randint(0, env.action_max, env.ZONE_NUM)
    # actions[:, 5:10, 0:10] = action1[0:10]
    # actions[:, 10:15, 10:20] = action2[10:20]
    # actions = torch.from_numpy(actions).float().to(env.device)
    #
    # random.seed(3074)
    # rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    # rand_idxs = torch.tensor(rand_list, device=env.device)
    # # 将单个环境的索引复制到每个批次
    # rand_idxs = rand_idxs.repeat(env.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组
    #
    # import time
    # start_time = time.time()
    # for i in range(1):
    #     env.reset(rand_idxs)
    #     ep_s = 0
    #     ep_r = torch.zeros((env.env_count,), device=env.device)
    #     while True:
    #
    #         s_, r, done, info = env.step(action=actions[:, ep_s, :])
    #         env.reset_contagious_OD()
    #         ep_s += 1
    #         ep_r += r
    #
    #
    #
    #         if done:
    #             ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_ado, ep_tdo = info.values()
    #             # env.render()
    #             env.render_one_region()
    #             # 总感染的峰值天数需要对每个批次进行计算
    #             daily_new_I = env.daily_new_I.cpu().numpy()
    #             daily_curr_I = env.simRes[:, :, :, 2].sum(dim=2)
    #             peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)
    #
    #             for batch_index in range(env.env_count):
    #                 total_new_I = sum(daily_new_I[batch_index])
    #                 print(f"总新增感染人数：{total_new_I}，现存感染最大峰值：{daily_curr_I[batch_index, peak_days[batch_index]]:.8f}")
    #                 print(
    #                     f"Batch {batch_index}, level {i} || reward: {ep_r[batch_index]:.4f}\t"
    #                     f"overload: {ep_overload[batch_index]:.4f}\t intensity: {ep_intensity[batch_index]:.2f}\t"
    #                     f"tdo: {ep_tdo[batch_index]:.4f}\t sdo: {ep_sdo[batch_index]:.4f}\t"
    #                     f"fdo: {ep_fdo[batch_index]:.2f}\t ado: {ep_ado[batch_index]:.2f}\t"
    #                     f"peak_day: {peak_days[batch_index]}"
    #                 )
    #
    #             break
    #
    # print("--- 耗时： %s seconds ---" % (time.time() - start_time))

    from config import args

    # args.I_obs_imperfect = True
    # args.I_obs_imperfect_down = 0.5
    args.device_name = 'cpu'
    env = EpidemicModelTensorVector(args, env_count=10)

    env.reset()

    while True:

        s_, r, done, info = env.step(action=torch.zeros(env.env_count, env.ZONE_NUM))
        env.reset_contagious_OD()

        if done:
            ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_ado, ep_tdo = info.values()
            # env.render()
            env.render_one_region()
            # 总感染的峰值天数需要对每个批次进行计算
            daily_new_I = env.daily_new_I.cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, 2].sum(dim=2)
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for batch_index in range(env.env_count):
                total_new_I = sum(daily_new_I[batch_index])
                print(f"总新增感染人数：{total_new_I}，现存感染最大峰值：{daily_curr_I[batch_index, peak_days[batch_index]]:.8f}")
                print(
                    f"Batch {batch_index}, level {0} || reward: {ep_r[batch_index]:.4f}\t"
                    f"overload: {ep_overload[batch_index]:.4f}\t intensity: {ep_intensity[batch_index]:.2f}\t"
                    f"tdo: {ep_tdo[batch_index]:.4f}\t sdo: {ep_sdo[batch_index]:.4f}\t"
                    f"fdo: {ep_fdo[batch_index]:.2f}\t ado: {ep_ado[batch_index]:.2f}\t"
                    f"peak_day: {peak_days[batch_index]}"
                )

            break