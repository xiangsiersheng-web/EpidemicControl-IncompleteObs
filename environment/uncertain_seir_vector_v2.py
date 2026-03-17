# -*- encoding: utf-8 -*-
# @Time    : 2024-10-10
# @File    : uncertain_seir_vector.py
import math
import os
import random
from datetime import datetime

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

# # 几种控制措施的定义
# action_to_u = torch.tensor([
#     [0.00, 0.00, 0.00],
#     [0.25, 0.00, 0.00],
#     [0.50, 0.00, 0.00],
# ], dtype=torch.float32)
action_to_u = torch.tensor([
    0, 0.25, 0.50, 0.75, 1.0
], dtype=torch.float32)


class EpidemicModel:
    S: int = 0
    E: int = 1
    I: int = 2
    R: int = 3
    Q: int = 4

    def __init__(self, args, env_count=20):
        args_dict = vars(args)
        self.reward_mode = args.reward_mode # 奖励函数的选择
        self.city = args.city # 城市名称
        self.R0 = args.R0   # 基本再生数决定beta
        self.env_count = int(env_count) # 并行环境数量
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device_name) # cuda or cpu
        self.action_to_u = action_to_u.to(self.device)
        data_dir = '../data/' if 'env_data_dir' not in args_dict else args.env_data_dir
        data_dir += self.city # 根据城市名称选择数据目录

        self.action_max = self.action_to_u.shape[0]
        if args.simulate_scale == 'district':
            self.OD = torch.from_numpy(np.load(data_dir + '/flow.npy')).float().to(self.device)
            self.POP = torch.from_numpy(np.load(data_dir + '/population.npy')).float().to(self.device)
        elif args.simulate_scale == 'community':
            data_dir += f'/community_{args.zone_num}'
            self.OD = torch.from_numpy(np.load(data_dir + '/flow.npy')).float().to(self.device)
            self.POP = torch.from_numpy(np.load(data_dir + '/population.npy')).float().to(self.device)
        self.ZONE_NUM = self.POP.shape[0]

        # 给矩阵增加一个第0维，用于后续的计算操作，和env_count对应
        self.OD = self.OD.unsqueeze(0).expand(env_count, -1, -1)
        self.original_OD = self.OD.clone() # 存储原始的流动矩阵
        self.contagious_OD = self.OD.clone() # 感染者的流动矩阵
        self.POP = self.POP.unsqueeze(0).expand(env_count, -1)
        self.TOTAL_POP = self.POP.sum(dim=-1)

        # 传染病相关参数（感染者移动比例、beta、潜伏期、恢复期）
        self.Pm = torch.tensor(0.8 if "ODE_Pm" not in args_dict else args.ODE_Pm, device=self.device)
        self.beta = 0.8 if "ODE_beta" not in args_dict else args.ODE_beta
        self.betas = torch.full((self.ZONE_NUM, ), self.beta, device=self.device)
        self.betas = self.betas.unsqueeze(0).expand(env_count, -1)
        self.beta_E_rate = torch.tensor(0.5, device=self.device)
        self.original_betas = self.betas.clone() # 存储原始的beta
        self.sigma = torch.tensor(1 / 3 if "ODE_sigma" not in args_dict else args.ODE_sigma, device=self.device)
        self.gamma = torch.tensor(1 / 7 if "ODE_gamma" not in args_dict else args.ODE_gamma, device=self.device)
        self.gamma_q = torch.tensor(1 / 7 if "ODE_gamma_q" not in args_dict else args.ODE_gamma, device=self.device)
        # E I 的检出率
        self.detect_E_rate = torch.tensor(0.60 if "ODE_detect_E_rate" not in args_dict else args.ODE_detect_E_rate, device=self.device)
        self.detect_I_rate = torch.tensor(0.95 if "ODE_detect_I_rate" not in args_dict else args.ODE_detect_I_rate, device=self.device)

        # 不同城市的医疗容量和ylim（绘图用）
        city_capacity = {'sz': 4e6, 'tokyo': 2.2e6, 'nyc': 2e6, 'sh': 5.5e6}
        self.capacity = torch.tensor(city_capacity[self.city], device=self.device)
        self.capacity = self.capacity if self.R0 == 'high' else self.capacity / 2
        self.capacity = 0.5 * self.capacity

        self.ylim = self.capacity.item() * 4

        # 模拟周期
        self.period = 120 if "ODE_period" not in args_dict else args.ODE_period

        # 状态的观测天数
        self.WINDOW_SIZE = 7 if "WINDOW_SIZE" not in args_dict else args.WINDOW_SIZE
            
        # 每个区域单步的观测维度
        self.local_obs_dim = 3 if "local_obs_dim" not in args_dict else args.local_obs_dim
        self.state_contain_toJ = False if "state_contain_toJ" not in args_dict else args.state_contain_toJ
        self.state_contain_action = False if "state_contain_action" not in args_dict else args.state_contain_action
        assert 3 + self.state_contain_toJ + self.state_contain_action == self.local_obs_dim, "local_obs_dim error"

        # 状态缩放尺度
        self.state_standard_scale = 1e4 if "state_standard_scale" not in args_dict else args.state_standard_scale

        # 是否使用reward shaping
        self.use_reward_shaping = False if "use_reward_shaping" not in args_dict else args.use_reward_shaping

        # 奖励函数的权重
        self.reward_weights = [1, 1, 1] if "reward_weights" not in args_dict else args.reward_weights
        self.local_reward_weight = 0.8 if "local_reward_weight" not in args_dict else args.local_reward_weight

        """ 不确定性1：不完全观测 """
        # TODO: 观测的不完全通过detect_E_rate和detect_I_rate进行控制
        # 是否对新增I进行不完全观测
        self.I_obs_imperfect = False if "I_obs_imperfect" not in args_dict else args.I_obs_imperfect
        self.I_obs_imperfect_up = 1.0 if "I_obs_imperfect_up" not in args_dict else args.I_obs_imperfect_up
        self.I_obs_imperfect_down = 0.3 if "I_obs_imperfect_down" not in args_dict else args.I_obs_imperfect_down
        # 是否基于不完全观测计算奖励
        # self.use_imperfect_calc_reward = False if "use_imperfect_calc_reward" not in args_dict else args.use_imperfect_calc_reward

        """ 不确定性2：突发异常事件 """
        # 异常时空行为，但该属性没有启用，异常事件由外部调用
        self.gather_to_some_region = False if "gather_to_some_region" not in args_dict else args.gather_to_some_region # I向某些区域聚集

        """ 不确定性3：传播参数变化波动 """
        # beta 变化导致的状态转移不确定
        self.use_beta_change = False if "use_beta_change" not in args_dict else args.use_beta_change # beta变化
        self.beta_matrix = None if "beta_matrix" not in args_dict else args.beta_matrix # beta变化矩阵
        self.original_beta_matrix = self.beta_matrix.clone() if self.beta_matrix is not None else None # 存储原始的beta变化矩阵
        self.state_contain_beta = False if "state_contain_beta" not in args_dict else args.state_contain_beta # 是否在状态中包含beta
        if self.state_contain_beta: raise NotImplementedError
        
        """ 不确定性4：动作执行效果的不确定 """
        # TODO：动作执行效果的不确定
        self.use_action_uncertainty = False if "use_action_uncertainty" not in args_dict else args.use_action_uncertainty # 动作执行效果的不确定

    def seed(self, seed=3047):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


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
        self.actions = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 2), device=self.device)

        # 初始化历史成本数据，每个成本数据维度为 (env_count, period)
        self.history_cost = {
            "reward": torch.zeros((self.env_count, self.period), device=self.device),
            "test_num": torch.zeros((self.env_count, self.period), device=self.device),
            "quara_num": torch.zeros((self.env_count, self.period), device=self.device),
            "local_reward": torch.zeros((self.env_count, self.period), device=self.device),
            "global_reward": torch.zeros((self.env_count, self.period), device=self.device),
        }

        # 初始化仿真状态，扩展为 (env_count, ZONE_NUM, 5)
        self.simState = torch.zeros((self.env_count, self.ZONE_NUM, 6), device=self.device).float()
        self.simState[:, :, 0] = self.POP

        # 设置初始感染者种子
        self.set_init_seed(rand_idxs=rand_idxs)

        # 初始化仿真结果数组，现在为四维数组：(env_count, period+1, ZONE_NUM, 5)
        self.simRes = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 6), device=self.device)
        self.simRes[:, 0, :, :] = self.simState

        # 初始化日常新发事件数组
        self.daily_new_E = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_new_I = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 记录每日从隔离仓室中恢复为R的数量，检测时应该排除掉这些人以及Q（这部分是已知恢复的、已隔离的人）
        self.daily_Q2R = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 随机化各个环境各个区域的检测率
        if self.I_obs_imperfect:
            raise NotImplementedError

        # 初始化不完全观测
        # self.imperfect_states = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 4), device=self.device)
        # self.imperfect_daily_new_I = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        # self._construct_imperfect_obs()

        # 记录每天到达区域J的人口数量
        self.POP_toJ = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.POP_toJ[:, 0, :] = (self.OD * self.POP.unsqueeze(-1)).sum(dim=1) # 初始化每天到达区域J的人口数量

        # 记录历史观测（减少重复计算）
        self.history_local_obs = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, self.local_obs_dim), device=self.device)

        # 构建初始观察窗口
        obs = self._get_obs() # shape(env_count, zone_num, window_size, local_obs_dim * 2)

        # 记录惩罚次数
        if self.use_reward_shaping:
            self.penalty_times = 0

        return obs

    def set_init_seed(self, init_infection=100, rand_idxs=None):
        """
        设置初始感染者种子。

        此函数用于：
        - 随机选择初始感染者位置。
        - 将初始感染者数量添加到初始状态中。
        """
        if rand_idxs is None:
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

    def obs_imperfect(self):
        """ 不完全观测：用于构造各区域的检测率和检出延迟，每次reset重置时应该调用 """
        # 1.各区域的检测率均匀分布在[down, up]之间
        up = self.I_obs_imperfect_up
        down = self.I_obs_imperfect_down
        self.I_detect_rate = torch.rand((self.env_count, self.ZONE_NUM), device=self.device) * (up - down) + down # shape: (env_count, zone_num)

        # 2.各区域的检测延迟，对于会被检测出的感染者，延迟时间分布在未来[0,4]
        day_cnt = 5
        delay_mean1 = self._delay_day_distribution(day_cnt, mean=1, std=1.5)
        delay_mean2 = self._delay_day_distribution(day_cnt, mean=2, std=1.5)

        # 2.1.初始化延迟分布张量，默认所有区域使用 delay_mean1
        self.delay_distribution = delay_mean1.unsqueeze(0).unsqueeze(0).expand(self.env_count, self.ZONE_NUM, day_cnt).clone() # shape: (env_count, zone_num, day)

        # 2.2.生成随机掩码
        random_mask = torch.rand((self.env_count, self.ZONE_NUM), device=self.device) < 0.3
        random_mask_expanded = random_mask.unsqueeze(-1).expand(-1, -1, day_cnt)
        full_delay_mean2 = delay_mean2.unsqueeze(0).unsqueeze(0).expand(self.env_count, self.ZONE_NUM, day_cnt).clone()
        # 将随机掩码对应的区域使用 delay_mean2
        self.delay_distribution[random_mask_expanded] = full_delay_mean2[random_mask_expanded]


    def _delay_day_distribution(self, day_cnt=5, mean=1, std=1.5):
        """ 生成一个长度为day_cnt的延迟分布，符合正态分布 """
        days = torch.arange(day_cnt + 1, dtype=torch.float32, device=self.device)
        cdf_values = 0.5 * (1 + torch.erf((days - 0.5 - mean) / (std * math.sqrt(2))))
        probabilities = torch.diff(cdf_values)
        probabilities /= probabilities.sum()
        return probabilities

    def _construct_imperfect_obs(self, std = 0.05):
        """ 根据检测率和检测延迟，构造不完全观测的I，应该在每天模拟结束后调用一次
        说明：不完全观测应该针对新增I，而不是现存I；同时要根据不完全观测I计算R
        """
        if not self.I_obs_imperfect:
            self.imperfect_states[:, self.day - 1, :, :] = self.simRes[:, self.day - 1, :, :] # shape: (env_count, zone_num, 4)
            self.imperfect_daily_new_I[:, self.day - 1, :] = self.daily_new_I[:, self.day - 1, :] # shape: (env_count, zone_num)
            return

        true_new_I = self.daily_new_I[:, self.day - 1, :].clone() # shape: (env_count, zone_num)
        # 为I添加噪声
        new_I_noise = true_new_I * (self.I_detect_rate + torch.randn_like(true_new_I, device=self.device) * std) # shape: (env_count, zone_num)
        new_I_noise = torch.clamp(new_I_noise, min=0)
        new_I_noise = new_I_noise.unsqueeze(-1) # shape: (env_count, zone_num, 1)
        distributed_new_I_noise = new_I_noise * self.delay_distribution # shape: (env_count, zone_num, day_cnt)
        # 将new_I_noise分布到未来几天
        day_cnt = distributed_new_I_noise.shape[-1]
        day_cnt = day_cnt if self.day-1+day_cnt <= self.period + 1 else self.period + 1 - self.day + 1
        self.imperfect_daily_new_I[:, self.day - 1:self.day - 1 + day_cnt, :] += torch.transpose(distributed_new_I_noise, 1, 2)[:, :day_cnt, :] # shape: (env_count, day_cnt, zone_num)

        dR = self.gamma * self.imperfect_states[:, self.day - 2, :, 2] # shape: (env_count, zone_num)
        dI = self.imperfect_daily_new_I[:, self.day - 1, :] - dR # shape: (env_count, zone_num)
        state = torch.zeros_like(self.simState) # shape: (env_count, zone_num, 4)

        state[:, :, 3] = self.imperfect_states[:, self.day - 2, :, 3] + dR # shape: (env_count, zone_num)
        state[:, :, 2] = self.imperfect_states[:, self.day - 2, :, 2] + dI # shape: (env_count, zone_num)
        state[:, :, 0] = self.POP - self.imperfect_states[:, self.day - 1, :, 2:].sum(dim=-1) # shape: (env_count, zone_num)

        # TODO:对state中较小的元素置为0
        # state[state < self.zero_threshold] = 0
        state = self._process_less_than_one(state)
        self.imperfect_states[:, self.day - 1, :, :] = state # shape: (env_count, zone_num, 4)

    def _process_less_than_one(self, x, threshold=1):
        # 创建与 x 相同形状的均匀分布随机数
        random_values = torch.rand_like(x)

        # 对小于 threshold 的元素，按概率决定是 0 或 1
        mask = x < threshold
        x[mask] = (x[mask] > random_values[mask]).float()

        return x

    def adjust_contagious_OD(self, regions = [0], increase_ratio = [0.5]):
        """ 人群聚集的模拟
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
        """
        self.contagious_OD = self.OD.clone()

    def adjust_OD(self, regions = [0], increase_ratio = [0.5], beta_increase_ratio = [0.5]):
        """ 人群聚集的模拟（考虑整体流动矩阵的聚集，以及聚集区域应该有beta升高的风险）
        此函数用于调整 OD
        :param regions: 要调整的区域
        :param increase_ratio: 要提高的流入比例
        :param beta_increase_ratio: 要提高的beta比例
        """
        # 1.确保regions和increase_ratio列表长度一致
        assert len(regions) == len(increase_ratio) == len(beta_increase_ratio), "regions, increase_ratio and beta_increase_ratio 列表长度需相等"

        # 2.调整OD
        self.OD = self.original_OD.clone()
        # 2.1.遍历每个需要调整的区域及其对应的增加比例
        for region, ratio in zip(regions, increase_ratio):
            # 增加指定区域的流入比例
            self.OD[:, :, region] *= (1 + ratio)

        # 2.2.重新归一化以保持每个区域的流出比例总和为1
        # 注意：这种计算较简洁，但并不能真正提高到预定的比例，会略低一点
        sum_along_dim2 = self.OD.sum(dim=2, keepdim=True)
        self.OD /= sum_along_dim2
        self.contagious_OD = self.OD.clone()

        # 3.调整beta
        if self.use_beta_change and self.beta_matrix is not None:
            # 需要调整beta_matrix
            self.beta_matrix = self.original_beta_matrix.clone()
            for region, ratio in zip(regions, beta_increase_ratio):
                self.beta_matrix[:, :, region] *= (1 + ratio)
        else:
            # 需要调整betas
            self.betas = self.original_betas.clone()
            for region, ratio in zip(regions, beta_increase_ratio):
                self.betas[region] *= (1 + ratio)

        # 输出提高比例
        for region in regions:
            print(
                f"Region {region}: from {self.original_OD[:, :, region].sum(dim=1).mean(dim=0)} to {self.OD[:, :, region].sum(dim=1).mean(dim=0)}")

    def reset_OD(self):
        """
        重置OD为原始的OD矩阵，beta为原始的betas or beta_matrix
        """
        self.OD = self.original_OD.clone()
        self.contagious_OD = self.OD.clone()
        if self.use_beta_change and self.beta_matrix is not None:
            self.beta_matrix = self.original_beta_matrix.clone()
        else:
            self.betas = self.original_betas.clone()


    def dynamic_beta_change(self, type = 'none', std = 0.1):
        """ 动态调整beta矩阵，用于模拟beta的变化

        根据参数调整对象的beta_matrix(shape: (env_count, period, zone_num))

        :param type: str, 'none'表示不调整，'spatial'表示按空间调整，'temporal'表示按时间调整，'spatiotemporal'表示按空间和时间调整
        :param std: float, 表示调整的std
         """
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
            noise = torch.randn(self.env_count, self.ZONE_NUM, device=self.device) * std
            beta_spatial_noise = beta_spatial * (1 + noise) # shape: (env_count, zone_num)
            self.beta_matrix += beta_spatial_noise.unsqueeze(1).expand(-1, self.period, -1)
        elif type == 'temporal':
            beta_temporal = torch.full((self.period, ), self.beta, device=self.device)
            self.beta_matrix = torch.zeros((self.env_count, self.period, self.ZONE_NUM)).to(self.device)
            noise = torch.randn(self.env_count, self.period, device=self.device) * std
            beta_temporal_noise = beta_temporal * (1 + noise) # shape: (env_count, period)
            self.beta_matrix += beta_temporal_noise.unsqueeze(2).expand(-1, -1, self.ZONE_NUM)
        elif type == 'spatiotemporal':
            beta_spatialtemporal = torch.full((self.period, self.ZONE_NUM), self.beta, device=self.device)
            self.beta_matrix = torch.zeros((self.env_count, self.period, self.ZONE_NUM)).to(self.device)
            noise = torch.randn(self.env_count, self.period, self.ZONE_NUM, device=self.device) * std
            beta_spatialtemporal_noise = beta_spatialtemporal * (1 + noise) # shape: (env_count, period, zone_num)
            self.beta_matrix += beta_spatialtemporal_noise
        self.original_beta_matrix = self.beta_matrix.clone()

    def _uncertain_action(self, u0, u1, std = 0.025):
        """ 动作的不确定性：影响系统的状态转移 """
        actual_u0 = u0 * (1 + std * torch.randn_like(u0))
        actual_u1 = u1 * (1 + std * torch.randn_like(u1))
        return actual_u0, actual_u1


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
        # 记录动作
        self.day += 1
        if action is None:
            # 每个区域有[p_test, p_quarantinue]
            action = torch.zeros((self.env_count, self.ZONE_NUM, 2), device=self.device)
        # 确保action是(env_count, ZONE_NUM)的Tensor
        assert action.shape == (self.env_count, self.ZONE_NUM, 2), "Action shape must be (env_count, ZONE_NUM)"
        self.actions[:, self.day - 2, :] = action # shape (env_count, ZONE_NUM, 2)

        # 如果beta是动态变化的，则根据天数更新beta
        if self.use_beta_change:
            self.betas = self.beta_matrix[:, self.day - 2, :]

        # 将action映射为u0 u1
        p_test, p_quara = self._action_to_u(action) # (env_count, ZONE_NUM)

        # 是否要对动作增加不确定性
        if self.use_action_uncertainty:
            # 如果要对动作增加不确定性，就给动作加上一个不确定，再执行
            actual_p_test, actual_p_quara = self._uncertain_action(p_test, p_quara)
        else:
            actual_p_test, actual_p_quara = p_test, p_quara

        # 仓室模型step
        contagious_infects = self.beta_E_rate * self.simState[:, :, 1] + self.Pm * self.simState[:, :, 2]  # (env_count, ZONE_NUM)

        # 计算传染病传播到每个区域的影响，使用contagious_OD的转置
        contagious_toJ = torch.bmm(self.OD.transpose(2, 1), contagious_infects.unsqueeze(-1)).squeeze(-1) # (env_count, ZONE_NUM)
        # 如果流出感染人数小于1，则概率转为0/1
        contagious_toJ = self._process_less_than_one(contagious_toJ)
        contagious_toJ = torch.clamp(contagious_toJ, min=0.0)

        all_toJ = torch.bmm(self.OD.transpose(2, 1), self.POP.unsqueeze(-1)).squeeze(-1) # (env_count, ZONE_NUM)

        contagious_ratio_toJ = contagious_toJ / all_toJ # (env_count, ZONE_NUM)
        contagious_ratio_toJ = torch.clamp(contagious_ratio_toJ, min=0.0)

        # 计算带有 betas 的 contagious_ratioToJ
        modified_ratio = contagious_ratio_toJ * self.betas # (env_count, ZONE_NUM)
        lam = torch.bmm(self.OD, modified_ratio.unsqueeze(-1)).squeeze(-1) # (env_count, ZONE_NUM)
        lam = torch.clamp(lam, min=0.0)

        # 计算actual_p_test, actual_p_quara带来的影响
        detect_E = actual_p_test * self.simState[:, :, 1] * self.detect_E_rate # (env_count, ZONE_NUM)
        detect_I = actual_p_test * self.simState[:, :, 2] * self.detect_I_rate # (env_count, ZONE_NUM)
        quarantine_E = actual_p_quara * detect_E # (env_count, ZONE_NUM)
        quarantine_I = actual_p_quara * detect_I # (env_count, ZONE_NUM)

        dS = -self.simState[:, :, 0] * lam
        dE = self.simState[:, :, 0] * lam - self.sigma * self.simState[:, :, 1] - quarantine_E
        dI = self.sigma * self.simState[:, :, 1] - self.gamma * self.simState[:, :, 2] - quarantine_I
        dQE = quarantine_E - self.sigma * self.simState[:, :, 3]
        dQI = quarantine_I + self.sigma * self.simState[:, :, 3] - self.gamma_q * self.simState[:, :, 4]
        dR = self.gamma * self.simState[:, :, 2] + self.gamma_q * self.simState[:, :, 4]

        # 更新状态
        dState = torch.stack([dS, dE, dI, dQE, dQI, dR], dim=2) # (env_count, ZONE_NUM, 5)

        # 计算新增的E I
        self.daily_new_E[:, self.day - 1, :] = self.simState[:, :, 0] * lam # (env_count, zone_num)
        self.daily_new_I[:, self.day - 1, :] = self.sigma * self.simState[:, :, 1] + self.sigma * self.simState[:, :, 3] # (env_count, zone_num)
        self.daily_Q2R[:, self.day - 1, :] = self.gamma_q * self.simState[:, :, 4] # (env_count, zone_num)
        self.daily_new_E[:, self.day - 1, :] = self._process_less_than_one(self.daily_new_E[:, self.day - 1, :])
        self.daily_new_I[:, self.day - 1, :] = self._process_less_than_one(self.daily_new_I[:, self.day - 1, :])
        self.daily_Q2R[:, self.day - 1, :] = self._process_less_than_one(self.daily_Q2R[:, self.day - 1, :])

        # 计算新增后再更新状态
        self.simState += dState
        self.simState[self.simState < 0] = 0
        # 人数小于1时，发生概率转移0/1
        self.simState = self._process_less_than_one(self.simState)
        self.simState[:, :, 0] = self.POP - self.simState[:, :, 1:].sum(dim=2) # 修正总人数
        self.simRes[:, self.day - 1, :, :] = self.simState
        # self._construct_imperfect_obs()

        # 构造state/obs
        obs = self._get_obs(quarantine_E, quarantine_I) # shape: (env_count, window_size, zone_num, local_obs_dim * 2)
            
        # 计算奖励（根据期望动作计算，而不是动作的真实效果actual_action）
        reward = self._reward_func(actual_p_test) # shape: (env_count, zone_num)

        info = {}
        done = False
        if self.day > self.period:
            # 计算指标
            ep_r = self.history_cost['reward'].sum(dim=1)  # (env_count,)
            total_infections = self.daily_new_I.sum(dim=[1,2])  # (env_count)
            total_test_num = self.history_cost['test_num'].sum(dim=1) # (env_count,)
            total_quarantine = self.history_cost['quara_num'].sum(dim=1) # (env_count,)
            score = torch.exp((20 * total_infections) / self.TOTAL_POP) + torch.exp((total_quarantine + 1/8 * total_test_num) / self.TOTAL_POP) # (env_count)
            print("reward:", ep_r, "total_infections:", total_infections, "total_test_num:", total_test_num, "total_quarantine:", total_quarantine)

            if self.use_reward_shaping:
                print("penalty_times:", self.penalty_times)

            info = {
                'ep_r': ep_r,
                'total_infections:': total_infections,
                'total_test_num:': total_test_num,
                'total_quarantine:': total_quarantine,
                # 'score': score,
            }

            done = True

        # 理论上done也要批次化，但对于这个环境，大家都是一起开始，一起结束的，所以done只是一个标量
        return obs, reward, done, info

    def _action_to_u(self, action):
        """将action（离散值）映射到两种控制措施上

        :param action: (env_count, ZONE_NUM, 2)
                两种控制措施都是越大，越严格
        """
        action = action.to(torch.long)
        u_p_test = self.action_to_u[action[:, :, 0]] # (env_count, ZONE_NUM)
        u_p_quara = self.action_to_u[action[:, :, 1]] # (env_count, ZONE_NUM)
        return u_p_test, u_p_quara

    def _get_history_padding(self, datas, day, window_size):
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
    
        return window.clone()

    def _get_obs(self, quarantine_E=None, quarantine_I=None):
        """获取观测值

        :return: (env_count, ZONE_NUM, window_size, local_obs_dim * 2)
        """
        # total q2r
        total_q2r = self.daily_Q2R.sum(dim=1) # shape: (env_count, zone_num)
        if quarantine_E is None:
            quarantine_E = torch.zeros_like(total_q2r)
            quarantine_I = torch.zeros_like(total_q2r)
        # # 归一化
        quarantine_E, quarantine_I, total_q2r = quarantine_E / self.POP, quarantine_I / self.POP, total_q2r / self.POP
        curr_obs = torch.stack([quarantine_E, quarantine_I, total_q2r], dim=-1) # shape: (env_count, zone_num, 3)
        if self.state_contain_action:
            raise NotImplementedError
            # # curr action
            # action = self.actions[:, self.day - 2, :].unsqueeze(-1).clone() # shape: (env_count, zone_num)
            # curr_obs = torch.cat([curr_obs, action], dim=-1)

        self.history_local_obs[:, self.day - 1, :, :] = curr_obs # shape: (env_count, zone_num, local_obs_dim)

        local_obs = self._get_history_padding(self.history_local_obs, self.day, self.WINDOW_SIZE) # shape: (env_count, WINDOW_SIZE, ZONE_NUM, local_obs_dim)
        global_obs = local_obs.mean(dim=2) # shape: (env_count, WINDOW_SIZE, local_obs_dim)
        global_obs_expanded = global_obs.unsqueeze(2).repeat(1, 1, local_obs.size(2), 1)
        combined_obs = torch.cat([local_obs, global_obs_expanded], dim=-1) # shape: (env_count, WINDOW_SIZE, ZONE_NUM, local_obs_dim * 2)

        return combined_obs

    def _reward_func(self, actual_p_test):
        """计算每个区域得到的奖励

        说明：每个区域都有自己的时间秩序度，但空间秩序度只出现在全局奖励上

        :return shape:(env_count, zone_num)
        """
        state = self.simRes[:, self.day - 1, :, :]

        # 1.感染成本（E， I， new）
        infections = state[:, :, 1] + state[:, :, 2]
        # new_infections = self.daily_new_E[:, self.day - 1, :] + self.daily_new_I[:, self.day - 1, :] # (env_count, zone_num)
        local_infe_cost = (20 * infections) ** 2 / self.POP # (env_count, zone_num)
        global_infe_cost = (20 * infections.sum(dim=1)) ** 2 / self.TOTAL_POP # (env_count,)

        # 2.检测成本 actual_p_test
        test_num = actual_p_test * (self.POP - state[:, :, [3,4]].sum(dim=2) - self.daily_Q2R.sum(dim=1)) # (env_count, zone_num)
        local_test_cost = test_num / self.POP # (env_count, zone_num)
        global_test_cost = test_num.sum(dim=1) / self.TOTAL_POP # (env_count,)

        # 3.隔离成本 Q
        local_quara_cost = state[:, :, [3,4]].sum(dim=2) / self.POP # (env_count, zone_num)
        global_quara_cost = state[:, :, [3,4]].sum(dim=[1,2]) / self.TOTAL_POP # (env_count,)

        # rw = self.reward_weights
        rw = [1, 1/8, 1]
        # 1.global reward
        global_reward = - (rw[0] * global_infe_cost + rw[1] * global_test_cost + rw[2] * global_quara_cost) # shape: (env_count,)
        global_reward_expaned = global_reward.unsqueeze(1).repeat(1, self.ZONE_NUM) # shape: (env_count, zone_num)

        # 2.local reward
        local_reward = -(rw[0] * local_infe_cost + rw[1] * local_test_cost + rw[2] * local_quara_cost) # shape: (env_count, zone_num)

        reward = self.local_reward_weight * local_reward + (1 - self.local_reward_weight) * global_reward_expaned # shape: (env_count, zone_num)

        if self.use_reward_shaping:
            raise NotImplementedError

        # 记录奖励，确保为每个批次独立记录
        self.history_cost['reward'][:, self.day - 2] = reward.mean(dim=-1)
        self.history_cost['test_num'][:, self.day - 2] = test_num.sum(dim=1)
        self.history_cost['quara_num'][:, self.day - 2] = state[:, :, [3,4]].sum(dim=[1,2])
        self.history_cost['local_reward'][:, self.day - 2] = local_reward.mean(dim=-1)
        self.history_cost['global_reward'][:, self.day - 2] = global_reward

        return reward

    def render(self, title='', fig_dir=None) :
        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        # 计算所有环境的感染总和并取平均
        infections = self.simRes[:, :, :, 2].sum(dim=2).mean(dim=0)  # (period+1,)
        plt.plot(infections.cpu(), label='Infections (avg)', color='orange', linewidth=2)

        # 绘制不完美观测
        I_imperfect = self.imperfect_states[:, :, :, 2].sum(dim=2).mean(dim=0)
        plt.plot(I_imperfect.cpu(), label='Infections (imperfect)', color='red', linewidth=2)

        plt.axhline(self.capacity.item(), ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, self.ylim)

        plt.legend(fontsize=8, loc='upper left')
        plt.twinx()

        # 计算所有批次的NPI平均值
        actions_avg = self.actions.mean(dim=0).mean(dim=1)  # (period+1,)
        plt.plot(actions_avg.cpu(), label='NPI (avg)', color='green', linewidth=2)

        # 绘制动作
        plt.plot(self.actions.mean(dim=0).cpu(), alpha=0.3, color='0.5')

        plt.ylim(-1, self.action_max)
        plt.yticks(range(0, self.action_max, 1))

        plt.title("Daily Current Infection (avg) (" + self.city + ")" if title == '' else title)

        plt.legend(fontsize=8, loc='upper right')

        # 如果 `fig_dir` 为 None，则直接展示图像
        if fig_dir is None:
            plt.show()
        else:
            # 确保输出目录存在
            os.makedirs(fig_dir, exist_ok=True)

            # 生成以日期时间为文件名的文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(fig_dir, f"{timestamp}.png")

            # 保存图像
            plt.savefig(file_path, bbox_inches='tight')
            print(f"Figure saved to {file_path}")

        # 关闭图形，释放内存
        plt.close()



    def render_one_env(self, title='', env_idx=0) :
        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        # 绘制不完美观测
        I_imperfect = self.imperfect_states[env_idx, :, :, 2].sum(dim=1)
        plt.plot(I_imperfect.cpu(), label='Infections (imperfect)', color='red', linewidth=2)

        # 计算环境的感染总和并取平均
        infections = self.simRes[env_idx, :, :, 2].sum(dim=1)  # (period+1,)
        plt.plot(infections.cpu(), label='Infections', color='orange', linewidth=2)

        plt.axhline(self.capacity.item(), ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, self.ylim)

        plt.legend(fontsize=8, loc='upper left')
        plt.twinx()

        # 计算所有区域的NPI平均值
        actions_avg = self.actions[env_idx, :, :].mean(dim=1)  # (period+1,)
        plt.plot(actions_avg.cpu(), label='NPI (avg)', color='green', linewidth=2)

        # 绘制动作
        plt.plot(self.actions[env_idx, :, :].cpu(), alpha=0.3, color='0.5')

        plt.ylim(-1, self.action_max)
        plt.yticks(range(0, self.action_max, 1))

        plt.title(f"Daily Current Infection (env idx={env_idx}) (" + self.city + ")" if title == '' else title)

        plt.legend(fontsize=8, loc='upper right')

        plt.show()

    def render_one_region(self, title='', env_idx=0, region_idx=0) :
        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        # 绘制不完美观测
        I_imperfect = self.imperfect_states[env_idx, :, region_idx, 2]
        plt.plot(I_imperfect.cpu(), label='Infections (imperfect)', color='red', linewidth=2)

        # 该区域的感染曲线
        infections = self.simRes[env_idx, :, region_idx, 2]  # (period+1,)
        plt.plot(infections.cpu(), label='Infections', color='orange', linewidth=2)

        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, 6.5e4)

        plt.legend(fontsize=14, loc='upper left')
        plt.twinx()

        # 计算NPI平均值
        actions_avg = self.actions[env_idx, :, region_idx].cpu()  # (period+1,)
        plt.plot(actions_avg.cpu(), label='NPI', color='green', linewidth=2)

        # 绘制动作
        plt.plot(self.actions[env_idx, :, region_idx].cpu(), alpha=0.3, color='0.5')

        # 输出动作变化的天数
        change_days = []
        for i in range(1, len(actions_avg) - 1):
            if actions_avg[i] != actions_avg[i - 1]:
                change_days.append(i)
        print("动作变化的天数",change_days)

        plt.ylim(-1, self.action_max)
        plt.yticks(range(0, self.action_max, 1))

        plt.title(f"Daily Current Infection (env idx={env_idx}, region idx={region_idx}) (" + self.city + ")" if title == '' else title)

        plt.legend(fontsize=14, loc='upper right')

        plt.show()

    def close(self):
        print('close')

def measure_run_time():
    # 1.测试耗时
    from config import args
    args.simulate_scale = 'community'
    args.zone_num = 643

    # args.device_name = 'cpu'
    env = EpidemicModel(args, env_count=2)
    actions = np.ones((env.env_count, 120, env.ZONE_NUM))
    actions = torch.from_numpy(actions).float().to(env.device)

    random.seed(3074)
    rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    rand_idxs = torch.tensor(rand_list, device=env.device)
    # 将单个环境的索引复制到每个批次
    rand_idxs = rand_idxs.repeat(env.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组

    import time

    start_time = time.time()
    for i in range(5):
        env.reset(rand_idxs)
        ep_s = 0
        while True:

            s_, r, done, info = env.step(action=actions[:, ep_s, :] * (i % 3))
            ep_s += 1

            if done:
                (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
                 ep_infections_rate, ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()

                # 总感染的峰值天数需要对每个批次进行计算
                peak_days = torch.argmax(env.simRes[:, :, :, 2].sum(dim=2), dim=1).tolist()  # (env_count,)

                for batch_index in range(env.env_count):
                    print(
                        f"Batch {batch_index}, level {i} || reward: {ep_r[batch_index]:.4f}\t"
                        f"overload: {ep_overload[batch_index]:.4f}\t intensity: {ep_intensity[batch_index]:.2f}\t"
                        f"tdo: {ep_tdo[batch_index]:.4f}\t sdo: {ep_sdo[batch_index]:.4f}\t"
                        f"fdo: {ep_fdo[batch_index]:.2f}\t ado: {ep_ado[batch_index]:.2f}\t"
                        f"peak_day: {peak_days[batch_index]}"
                    )

                break

    print("--- 耗时： %s seconds ---" % (time.time() - start_time))

def test_consistency():
    # 2.测试一致性 （应该是十分一致的）
    from config import args
    # args.I_obs_imperfect = True
    # args.I_obs_imperfect_down = 0.5
    args.device_name = 'cpu'
    env = EpidemicModel(args, env_count=10)
    actions = np.ones((env.env_count, 120, env.ZONE_NUM)) * 2
    np.random.seed(3047)
    action1 = np.random.randint(0, env.action_max, env.ZONE_NUM)
    np.random.seed(305)
    action2 = np.random.randint(0, env.action_max, env.ZONE_NUM)
    actions[:, 5:10, 0:10] = action1[0:10]
    actions[:, 10:15, 10:20] = action2[10:20]
    actions = torch.from_numpy(actions).float().to(env.device)

    random.seed(3074)
    rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    rand_idxs = torch.tensor(rand_list, device=env.device)
    # 将单个环境的索引复制到每个批次
    rand_idxs = rand_idxs.repeat(env.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组

    import time
    start_time = time.time()
    for i in range(1):
        env.reset(rand_idxs)
        ep_s = 0
        ep_r = torch.zeros((env.env_count,), device=env.device)
        while True:

            s_, r, done, info = env.step(action=actions[:, ep_s, :])
            env.reset_contagious_OD()
            ep_s += 1
            ep_r += r



            if done:
                (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
                 ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()
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
                        f"Batch {batch_index}, level {i} || reward: {ep_r[batch_index]:.4f}\t"
                        f"overload: {ep_overload[batch_index]:.4f}\t intensity: {ep_intensity[batch_index]:.2f}\t"
                        f"tdo: {ep_tdo[batch_index]:.4f}\t sdo: {ep_sdo[batch_index]:.4f}\t"
                        f"fdo: {ep_fdo[batch_index]:.2f}\t ado: {ep_ado[batch_index]:.2f}\t"
                        f"peak_day: {peak_days[batch_index]}"
                    )

                break

    print("--- 耗时： %s seconds ---" % (time.time() - start_time))


def test_control_timely():
    from config import args

    # torch.manual_seed(3047)
    args.simulate_scale = 'community'
    args.zone_num = 643
    # args.ODE_zero_threshold = 0.05

    # 不完全观测
    # args.I_obs_imperfect = True
    args.I_obs_imperfect_down = 0.5

    # 聚集行为
    # args.gather_to_some_region = True
    regions = [0]
    increase_ratio = [10]
    beta_increase_ratio = [3]

    # 动态变化的beta
    # args.use_beta_change = True
    args.env_beta_change_rule = 'spatial'

    # 动作的不确定
    # args.use_action_uncertainty = True

    # args.device_name = 'cuda:0'
    args.device_name = 'cpu'
    device = torch.device(args.device_name)
    env = EpidemicModel(args, env_count=2)
    env.seed(3047)

    rand_list = [0 for _ in range(100)]
    rand_idxs = torch.tensor(rand_list, device=env.device)
    # 将单个环境的索引复制到每个批次
    rand_idxs = rand_idxs.repeat(env.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组

    env.reset(rand_idxs=rand_idxs)

    # 动态变化的beta
    if env.use_beta_change:
        env.dynamic_beta_change(type=args.env_beta_change_rule)

    while True:

        if env.day == 45 and env.gather_to_some_region:
            env.adjust_OD(regions=regions, increase_ratio=increase_ratio, beta_increase_ratio=beta_increase_ratio)

        action = torch.ones(env.env_count, env.ZONE_NUM).to(device)
        action *= 0
        if env.day > 10:
            action *= 2
        action[:, 0] = 4
        s_, r, done, info = env.step(action=action)

        if env.day == 50:
            env.reset_OD()

        if done:
            (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
             ep_infections_rate, ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()
            env.render()
            env.render_one_region()
            # 总感染的峰值天数需要对每个批次进行计算
            daily_new_I = env.daily_new_I.cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, 2].sum(dim=2)
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"总新增感染人数：{total_new_I}，现存感染最大峰值：{daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")
                print(
                    f"Batch {env_idx}, level {0} || reward: {ep_r[env_idx]:.4f}\t"
                    f"overload: {ep_overload[env_idx]:.4f}\t intensity: {ep_intensity[env_idx]:.2f}\t"
                    f"tdo: {ep_tdo[env_idx]:.4f}\t sdo: {ep_sdo[env_idx]:.4f}\t"
                    f"fdo: {ep_fdo[env_idx]:.2f}\t ado: {ep_ado[env_idx]:.2f}\t"
                    f"peak_day: {peak_days[env_idx]}"
                )

            break


if __name__ == '__main__':
    # measure_run_time()
    # exit(0)
    # test_control_timely()
    # exit(0)
    from config import args


    args.simulate_scale = 'community'
    args.zone_num = 643

    # args.ODE_period = 200

    # 增加观测维度
    # args.state_contain_action = True
    args.local_obs_dim = 3

    # 不完全观测
    # args.I_obs_imperfect = True
    args.I_obs_imperfect_down = 1.0

    # 聚集行为
    # args.gather_to_some_region = True
    regions = [0]
    increase_ratio = [10]
    beta_increase_ratio = [3]

    # 动态变化的beta
    # args.use_beta_change = True
    args.env_beta_change_rule = 'spatial'

    # 动作的不确定
    # args.use_action_uncertainty = True


    # args.device_name = 'cuda:0'
    args.device_name = 'cpu'
    device = torch.device(args.device_name)

    args.ODE_detect_E_rate = 0.8
    args.ODE_detect_I_rate = 1.0

    env = EpidemicModel(args, env_count=2)
    env.seed(3047)
    env.reset()

    # 动态变化的beta
    if env.use_beta_change:
        env.dynamic_beta_change(type=args.env_beta_change_rule)


    while True:

        if env.day == 45 and env.gather_to_some_region:
            env.adjust_OD(regions=regions, increase_ratio=increase_ratio, beta_increase_ratio=beta_increase_ratio)

        action=torch.ones(env.env_count, env.ZONE_NUM, 2).to(device)
        action[:, :, 0] = 3
        action[:, :, 1] = 4
        s_, r, done, info = env.step(action=action)

        if env.day == 50:
            env.reset_OD()

        if done:
            # (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
            #  ep_infections_rate, ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()
            # env.render()
            # env.render_one_region()
            # 总感染的峰值天数需要对每个批次进行计算
            daily_new_I = env.daily_new_I.sum(dim=-1).cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, 2].sum(dim=2)
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"总新增感染人数：{total_new_I}，现存感染最大峰值：{daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")

            break