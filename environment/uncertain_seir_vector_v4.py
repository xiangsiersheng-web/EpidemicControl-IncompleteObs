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


action_to_u0 = torch.tensor([
    0, 0.05, 0.20, 0.5, 1.0
    # 0, 0.2, 0.5, 1.0
], dtype=torch.float32)
action_to_u1 = torch.tensor([
    0, 0.2, 1.0
    # 0, 1.0
], dtype=torch.float32)


class EpidemicModel:
    S : int = 0
    E_undetected : int = 1
    E_detected : int = 2
    I_undetected : int = 3
    I_detected : int = 4
    I_reported : int = 5
    QE : int = 6
    QI : int = 7
    R : int = 8

    E = [E_undetected, E_detected, QE]
    I = [I_undetected, I_detected, I_reported, QI]
    can_isolated = [E_detected, I_reported, I_detected]

    def __init__(self, args, env_count=20, is_evaluation=False):
        args_dict = vars(args)
        self.city = args.city # 城市名称
        self.env_count = int(env_count) # 并行环境数量
        self.is_evaluation = is_evaluation
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device_name) # cuda or cpu
        # self.action_to_u = action_to_u.to(self.device)
        self.action_to_u0 = action_to_u0.to(self.device)
        self.action_to_u1 = action_to_u1.to(self.device)
        self.action_dim = self.action_to_u0.size(0) * self.action_to_u1.size(0)

        data_dir = '../data/' if 'env_data_dir' not in args_dict else args.env_data_dir
        data_dir += self.city # 根据城市名称选择数据目录

        self.action_max = max(self.action_to_u0.shape[0], self.action_to_u1.shape[0])
        if args.simulate_scale == 'district':
            self.OD = torch.from_numpy(np.load(data_dir + '/flow.npy')).float().to(self.device)
            self.POP = torch.from_numpy(np.load(data_dir + '/population.npy')).float().to(self.device)
        elif args.simulate_scale == 'community':
            data_dir += f'/community_{args.zone_num}'
            self.OD = torch.from_numpy(np.load(data_dir + '/flow.npy')).float().to(self.device)
            self.POP = torch.from_numpy(np.load(data_dir + '/population.npy')).float().to(self.device)
        self.OD = self.OD / self.OD.sum(dim=-1, keepdim=True)
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
        self.R0 = 'high' if "R0" not in args_dict else args.R0
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
        # 检测资源函数的指数取值
        self.detection_efficiency_exp_param = torch.tensor(0.6 if "detection_efficiency_exp_param" not in args_dict else args.detection_efficiency_exp_param, device=self.device)
        # I的上报率
        self.report_I_rate0 = 1.0 if "ODE_report_I_rate" not in args_dict else args.ODE_report_I_rate

        # 每天输入的病例
        self.daily_imported_cases = 0 if "daily_imported_cases" not in args_dict else args.daily_imported_cases
        # 是否延迟隔离
        self.use_delay_quara = False if "use_delay_quara" not in args_dict else args.use_delay_quara

        # 不同城市的医疗容量和ylim（绘图用）
        city_capacity = {'sz': 4e6, 'tokyo': 2.2e6, 'nyc': 2e6, 'sh': 5.5e6}
        self.capacity = torch.tensor(city_capacity[self.city], device=self.device)
        # self.capacity = self.capacity
        self.capacity = 0.5 * self.capacity

        self.ylim = self.capacity.item() * 4

        # 模拟周期
        self.period = 120 if "ODE_period" not in args_dict else args.ODE_period

        # 状态的观测天数
        self.WINDOW_SIZE = 7 if "WINDOW_SIZE" not in args_dict else args.WINDOW_SIZE
            
        # 每个区域单步的观测维度
        self.local_obs_dim = 2 if "local_obs_dim" not in args_dict else args.local_obs_dim
        self.state_contain_detected = False if "state_contain_detected" not in args_dict else args.state_contain_detected
        self.state_contain_Q = False if "state_contain_Q" not in args_dict else args.state_contain_Q
        self.state_contain_R = False if "state_contain_R" not in args_dict else args.state_contain_R
        self.state_contain_detect_rate = False if "state_contain_detect_rate" not in args_dict else args.state_contain_detect_rate
        assert 2 + 2 * self.state_contain_detected + self.state_contain_Q + self.state_contain_R + self.state_contain_detect_rate == self.local_obs_dim, "local_obs_dim error"

        # 是否使用reward shaping
        self.use_reward_shaping = False if "use_reward_shaping" not in args_dict else args.use_reward_shaping

        # 奖励函数的权重
        self.reward_weights = [1, 1, 1] if "reward_weights" not in args_dict else args.reward_weights
        self.cost_ratio_t2q = 1/8 if "cost_ratio_test_to_quarantine" not in args_dict else args.cost_ratio_test_to_quarantine
        self.local_reward_weight = 1.0 if "local_reward_weight" not in args_dict else args.local_reward_weight
        # 是否停用奖励函数的责任分发
        self.disable_response_distribution = False if "disable_response_distribution" not in args_dict else args.disable_response_distribution

        """ 不确定性1：不完全观测 """
        # 观测的不完全目前是根据上报率做的不完全观测
        # 是否对新增I进行不完全观测
        self.use_obs_imperfect = False if "use_obs_imperfect" not in args_dict else args.use_obs_imperfect
        self.obs_imperfect_up = 1.0 if "obs_imperfect_up" not in args_dict else args.obs_imperfect_up
        self.obs_imperfect_down = 0.1 if "obs_imperfect_down" not in args_dict else args.obs_imperfect_down
        if self.obs_imperfect_down > self.obs_imperfect_up:
            raise ValueError("obs_imperfect_down > obs_imperfect_up", self.obs_imperfect_down, self.obs_imperfect_up)
        self.mask_rate_down = None if "mask_rate_down" not in args_dict else args.mask_rate_down
        self.mask_rate_up = None if "mask_rate_up" not in args_dict else args.mask_rate_up
        self.mask_duration_down = None if "mask_duration_down" not in args_dict else args.mask_duration_down
        self.mask_duration_up = None if "mask_duration_up" not in args_dict else args.mask_duration_up

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
        # 动作执行效果的不确定
        self.use_action_uncertainty = False if "use_action_uncertainty" not in args_dict else args.use_action_uncertainty # 动作执行效果的不确定

    def seed(self, seed=3047):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
        # 初始化动作数组
        self.actions = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 初始化历史成本数据，每个成本数据维度为 (env_count, period)
        self.history_cost = {
            "reward": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "test_num": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "quara_num": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "quara_days": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "local_reward": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "global_reward": torch.zeros((self.env_count, self.period), device=self.device),
            'local_infe_cost': torch.zeros((self.env_count, self.period), device=self.device),
            "local_test_cost": torch.zeros((self.env_count, self.period), device=self.device),
            "local_quara_cost": torch.zeros((self.env_count, self.period), device=self.device),
            "global_infe_cost": torch.zeros((self.env_count, self.period), device=self.device),
            "global_test_cost": torch.zeros((self.env_count, self.period), device=self.device),
            "global_quara_cost": torch.zeros((self.env_count, self.period), device=self.device),
        }

        # 初始化仿真状态，扩展为 (env_count, ZONE_NUM, 9) S E_undetected E_detected I_undetected I_detected I_reported QE QI R
        self.simState = torch.zeros((self.env_count, self.ZONE_NUM, 9), device=self.device).float()
        self.simState[:, :, 0] = self.POP

        # 设置初始感染者种子
        self.set_init_seed(rand_idxs=rand_idxs)

        # 初始化仿真结果数组，现在为四维数组：(env_count, period+1, ZONE_NUM, 9)
        self.simRes = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 9), device=self.device)
        self.simRes[:, 0, :, :] = self.simState

        # 初始化日常新发事件数组
        # 需要重建的信息2：每日新增E
        self.daily_new_E = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_new_E[:, 0, :] = self.simState[:, :, self.E_undetected]
        self.daily_new_I = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_new_Q = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_new_report = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 记录每日检测隔离
        self.daily_test_num = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_quara_num = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 记录延迟隔离
        self.daily_delay_quara_num = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 记录各区域感染者导致的新增E（为各区域的奖励计算服务）
        self.daily_new_E_self_cause = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 记录每日从已知的I、隔离仓室中恢复为R的数量，帮助判断恢复者趋势
        self.daily_known2R = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_detect_E = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)  # 已知的新增E
        self.daily_detect_I = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device) # 已知的新增I
        self.daily_estimate_new_EI = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # 随机化各个环境各个区域的检测率
        if self.use_obs_imperfect:
            self.obs_imperfect()

        # 观测缺失率记录
        self.obs_missing_total_days = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)  # 观测缺失天数
        self.has_been_observed = torch.zeros((self.env_count, self.ZONE_NUM), dtype=torch.bool, device=self.device) # 是否被观测

        # 记录历史观测（减少重复计算）
        self.history_local_obs = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, self.local_obs_dim), device=self.device)

        # 记录重建信息（可选，默认为空）
        self.rebuild_states = []

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
            # 方式一，随机化
            # random.seed(3074)
            rand_list = [random.randint(0, self.ZONE_NUM - 1) for _ in range(init_infection)]
            rand_idxs = torch.tensor(rand_list, device=self.device)
            # 将单个环境的索引复制到每个批次
            rand_idxs = rand_idxs.repeat(self.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组

            # # 方式二，选择人数最多的100个区域
            # rand_idxs = torch.argsort(self.POP, dim=-1, descending=True)[:, :init_infection]
            # rand_idxs = rand_idxs.to(self.device)



        # 初始化一个batch的计数器
        counts = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)

        # 使用scatter_add_进行并行累加，更新每个环境的感染者数量
        counts = counts.scatter_add_(1, rand_idxs, torch.ones_like(rand_idxs, dtype=torch.float))

        # 更新self.simState中对应索引的第一、二列
        self.simState[:, :, self.E_undetected] += counts
        self.simState[:, :, self.S] -= counts

    def _imported_cases(self, case_num):
        """输入外来病例"""
        # 生成符合正态分布的case_num_new，确保在0-2*case_num之间
        while True:
            sampled_value = random.gauss(case_num, case_num/2)
            if 0 <= sampled_value <= 2*case_num:
                case_num_new = round(sampled_value)
                break

        if case_num_new == 0:
            return

        case_num = case_num_new
        rand_list = [random.randint(0, self.ZONE_NUM - 1) for _ in range(case_num)]
        rand_idxs = torch.tensor(rand_list, device=self.device)
        # 将单个环境的索引复制到每个批次
        rand_idxs = rand_idxs.repeat(self.env_count, 1)  # 重复env_count次形成(env_count, init_infection)的数组

        # 初始化一个batch的计数器
        counts = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)

        # 使用scatter_add_进行并行累加，更新每个环境的感染者数量
        counts = counts.scatter_add_(1, rand_idxs, torch.ones_like(rand_idxs, dtype=torch.float))

        # 更新self.simState中对应索引的第一、二列
        self.simState[:, :, self.E_undetected] += counts
        self.simState[:, :, self.S] -= counts


    def obs_imperfect(self):
        """ 不完全观测：用于构造各区域的检测率和检出延迟，每次reset重置时应该调用 """
        # 1.各区域的检测率均匀分布在[down, up]之间
        up = self.obs_imperfect_up
        down = self.obs_imperfect_down
        self.I_real_report_rate = torch.rand((self.env_count, self.ZONE_NUM), device=self.device) * (up - down) + down # shape: (env_count, zone_num)



    def _delay_day_distribution(self, day_cnt=5, mean=1, std=1.5):
        """ 生成一个长度为day_cnt的延迟分布，符合正态分布 """
        days = torch.arange(day_cnt + 1, dtype=torch.float32, device=self.device)
        cdf_values = 0.5 * (1 + torch.erf((days - 0.5 - mean) / (std * math.sqrt(2))))
        probabilities = torch.diff(cdf_values)
        probabilities /= probabilities.sum()
        return probabilities


    def _add_noise(self, x, std = 0.05):
        """增加高斯噪声"""
        noise = 1 + torch.randn_like(x, device=self.device) * std
        noise = torch.clamp(noise, min=0, max=1)
        x = x * noise
        return x

    def _process_less_than_one(self, x, threshold=0.01):
        if threshold != 0:
            # # 创建与 x 相同形状的均匀分布随机数
            # random_values = torch.rand_like(x)
            #
            # # 对小于 threshold 的元素，按概率决定是 0 或 1
            # mask = x < threshold
            # x[mask] = (x[mask] > random_values[mask]).float()

            # 直接抹掉
            x[x < threshold] = 0
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

    def _uncertain_action(self, u0, u1, std = 0.1):
        """ 动作的不确定性：影响系统的状态转移 """
        actual_u0 = u0 * (1 + std * torch.randn_like(u0))
        actual_u1 = u1 * (1 + std * torch.randn_like(u1))
        actual_u0 = torch.clamp(actual_u0, min=0, max=1)
        actual_u1 = torch.clamp(actual_u1, min=0, max=1)
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
        self.day += 1
        # 输入外来病例（其实就是选中一些人，从S变为E，并没有改变总人口）
        self._imported_cases(self.daily_imported_cases)

        # 记录动作
        if action is None:
            action = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)
        # 确保action是(env_count, ZONE_NUM)的Tensor
        assert action.shape == (self.env_count, self.ZONE_NUM), "Action shape must be (env_count, ZONE_NUM, 2)"
        self.actions[:, self.day - 2, :] = action # shape (env_count, ZONE_NUM)

        # 如果beta是动态变化的，则根据天数更新beta
        if self.use_beta_change:
            self.betas = self.beta_matrix[:, self.day - 2, :]

        # 将action映射为u0 u1
        p_test, p_quara = self._action_to_u(action) # (env_count, ZONE_NUM)

        """是否要对动作增加不确定性"""
        if self.use_action_uncertainty:
            # 如果要对动作增加不确定性，就给动作加上一个不确定，再执行
            actual_p_test = self._add_noise(p_test, std=0.1)
            actual_p_quara = self._add_noise(p_quara, std=0.1)
            # actual_p_test, actual_p_quara = self._uncertain_action(p_test, p_quara)
        else:
            # 简单的动作扰动，有利于训练
            actual_p_test = self._add_noise(p_test, std=0.05)
            # actual_p_test = p_test
            actual_p_quara = p_quara

        """观测不完全也即给上报率加一个扰动"""
        if self.use_obs_imperfect:
            actual_p_report = self.report_I_rate0 * torch.ones(self.env_count, self.ZONE_NUM, device=self.device)

            # todo: 是否加上这个随机，如果加上，即便已知区域，其上报率也会未知
            # I_real_report_rate: 是一个分布在0.1-1.0的均匀分布，模拟观测不完全的影响（也即观测人数少于实际人数）
            actual_p_report = self._add_noise(self.report_I_rate0 * self.I_real_report_rate, std=0.1)

            mask = self._random_mask()
            actual_p_report[mask] = 0

            self.obs_missing_total_days += (mask & ~self.has_been_observed).to(torch.int32)
            self.has_been_observed |= (~mask)
        else:
            actual_p_report = self.report_I_rate0

        """3 step: 传播动力学；主动上报；检测-隔离"""
        # 1.传染导致的系统状态转移
        # 仓室模型step
        contagious_infects = (self.beta_E_rate * self.simState[:, :, [self.E_undetected, self.E_detected]].sum(dim=-1)
                    + self.Pm * self.simState[:, :, [self.I_undetected, self.I_detected, self.I_reported]].sum(dim=-1))  # (env_count, ZONE_NUM)
        # 计算传染病传播到每个区域的影响，使用contagious_OD的转置
        contagious_OD = self.OD * contagious_infects.unsqueeze(-1)  # (env_count, ZONE_NUM, ZONE_NUM)
        contagious_toJ = contagious_OD.sum(dim=1)  # (env_count, ZONE_NUM)
        contagious_OD_ratio = contagious_OD / (contagious_toJ.unsqueeze(1) + 1e-12)  # 列归一化，表示i到达j的感染者占到达j总感染者的比例
        # contagious_toJ = torch.bmm(self.OD.transpose(2, 1), contagious_infects.unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)
        # 如果流出感染人数小于1，则概率转为0/1
        # contagious_toJ = self._process_less_than_one(contagious_toJ)
        contagious_toJ = torch.clamp(contagious_toJ, min=0.0)

        all_toJ = torch.bmm(self.OD.transpose(2, 1), self.POP.unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)

        contagious_ratio_toJ = contagious_toJ / all_toJ  # (env_count, ZONE_NUM)
        contagious_ratio_toJ = torch.clamp(contagious_ratio_toJ, min=0.0)

        # 计算带有 betas 的 contagious_ratioToJ
        modified_ratio = contagious_ratio_toJ * self.betas  # (env_count, ZONE_NUM)
        lam = torch.bmm(self.OD, modified_ratio.unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)
        lam = torch.clamp(lam, min=0.0)

        # 计算传染导致的系统状态转移
        S2Eun = self.simState[:, :, self.S] * lam
        Eun2Iun = self.sigma * self.simState[:, :, self.E_undetected]
        Ede2Ide = self.sigma * self.simState[:, :, self.E_detected]
        Iun2R = self.gamma * self.simState[:, :, self.I_undetected]
        Ide2R = self.gamma * self.simState[:, :, self.I_detected]
        Ire2R = self.gamma * self.simState[:, :, self.I_reported]
        QE2QI = self.sigma * self.simState[:, :, self.QE]
        QI2R = self.gamma_q * self.simState[:, :, self.QI]

        dS = -S2Eun
        dE_undetected = S2Eun - Eun2Iun
        dE_detected = - Ede2Ide
        dI_undetected = Eun2Iun - Iun2R
        dI_detected = Ede2Ide - Ide2R
        dI_reported = - Ire2R
        dQE = - QE2QI
        dQI = QE2QI - QI2R
        dR = Iun2R + Ide2R + Ire2R + QI2R

        # 更新状态
        dState = torch.stack([dS, dE_undetected, dE_detected, dI_undetected, dI_detected, dI_reported, dQE, dQI, dR],
            dim=2)  # (env_count, ZONE_NUM, 9)
        self.simState += dState

        # 计算新增后再更新状态（为奖励函数的感染成本计算服务）
        # 将新增的E返回给各个区域
        S_toJ = torch.bmm(self.OD.transpose(2, 1), self.simState[:, :, self.S].unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)
        new_E_inJ = S_toJ * modified_ratio  # (env_count, ZONE_NUM) new_E_inJ.sum() == S2Eun.sum()，表示新增E还留在j区域，也即未返回的状态
        new_E_backI = (new_E_inJ.unsqueeze(1) * contagious_OD_ratio).sum(dim=-1)  # (env_count, ZONE_NUM)
        self.daily_new_E_self_cause[:, self.day - 1, :] = new_E_backI

        # 2. 主动上报导致一次状态转移
        Iun2Ire = actual_p_report * self.simState[:, :, self.I_undetected]
        self.simState[:, :, self.I_undetected] += - Iun2Ire
        self.simState[:, :, self.I_reported] += Iun2Ire
        self.simState[self.simState < 0] = 0
        assert (self.simState[:, :, :] >= 0).all(), "all state must be non-negative"
        self.daily_new_report[:, self.day - 1, :] = Iun2Ire

        # 3.1.状态转移-检测
        # 当前检测率下，受检的EI
        test_valid_num = actual_p_test * self.POP / self._cal_detection_scale()
        # test_num = torch.floor(test_num)
        test_num_rate = test_valid_num / (self.simState[:, :, [self.E_undetected, self.I_undetected]].sum(dim=-1) + 1e-8)
        test_num_rate = torch.clamp(test_num_rate, min=0, max=1)
        # test_num_rate = actual_p_test
        new_detect_E = test_num_rate * self.simState[:, :, self.E_undetected] * self.detect_E_rate  # (env_count, ZONE_NUM)
        new_detect_I = test_num_rate * self.simState[:, :, self.I_undetected] * self.detect_I_rate
        self.simState[:, :, self.E_undetected] -= new_detect_E
        self.simState[:, :, self.I_undetected] -= new_detect_I
        self.simState[:, :, self.E_detected] += new_detect_E
        self.simState[:, :, self.I_detected] += new_detect_I
        self.simState[self.simState < 0] = 0
        if not (self.simState[:, :, :] >= 0).all():
            print("all state must be non-negative")
        assert (self.simState[:, :, :] >= 0).all(), "all state must be non-negative"
        self.daily_detect_E[:, self.day - 1, :] = new_detect_E
        self.daily_detect_I[:, self.day - 1, :] = new_detect_I

        # 3.2.状态转移-隔离
        # 隔离的总人数 / 待隔离的人数
        # todo：确认今天计划隔离人数，并将计划数目分散到未来1-2天
        quara_plan_num = torch.min(actual_p_quara * self.POP, self.simState[:, :, self.can_isolated].sum(dim=-1))
        self.__delay_quara(quara_plan_num)
        # # 同时用今天可隔离人数，确定今日的隔离率
        quara_num_rate = self.daily_delay_quara_num[:, self.day - 1, :] / (self.simState[:, :, self.can_isolated].sum(dim=-1) + 1e-12)
        # quara_num_rate = quara_plan_num / (self.simState[:, :, self.can_isolated].sum(dim=-1) + 1e-12)
        quara_num_rate = torch.clamp(quara_num_rate, min=0, max=1)
        # quara_num_rate = actual_p_quara
        quarantine_E1 = quara_num_rate * self.simState[:, :, self.E_detected]  # (env_count, ZONE_NUM)
        quarantine_I1 = quara_num_rate * self.simState[:, :, self.I_detected]
        quarantine_I2 = quara_num_rate * self.simState[:, :, self.I_reported]
        self.simState[:, :, self.E_detected] -= quarantine_E1
        self.simState[:, :, self.I_detected] -= quarantine_I1
        self.simState[:, :, self.I_reported] -= quarantine_I2
        self.simState[:, :, self.QE] += quarantine_E1
        self.simState[:, :, self.QI] += quarantine_I1 + quarantine_I2
        self.simState[self.simState < 0] = 0
        assert (self.simState[:, :, :] >= 0).all(), "all state must be non-negative"

        # 记录新增的E I
        self.daily_new_E[:, self.day - 1, :] = S2Eun  # (env_count, zone_num)
        self.daily_new_I[:, self.day - 1, :] = Eun2Iun + Ede2Ide + QE2QI  # (env_count, zone_num)
        self.daily_known2R[:, self.day - 1, :] = Ire2R + Ide2R + QI2R  # (env_count, zone_num)
        self.daily_new_Q[:, self.day - 1, :] = quarantine_E1 + quarantine_I1 + quarantine_I2

        # 对状态中的较小值进行修正
        self.simState = self._process_less_than_one(self.simState)  # 人数小于1时，发生概率转移0/1
        self.simState[:, :, 0] = self.POP - self.simState[:, :, 1:].sum(dim=2)  # 修正总人数
        self.simState[self.simState < 0] = 0
        assert (self.simState[:, :, :] >= 0).all(), f"all state must be non-negative in {self.day}"
        self.simRes[:, self.day - 1, :, :] = self.simState.clone()

        # if self.day in range(10, 60, 10):
        #     self.__cal_pearsonr()

        # 构造state/obs
        obs = self._get_obs() # shape: (env_count, window_size, zone_num, local_obs_dim * 2)
            
        # 计算奖励（根据动作的真实效果actual_action）
        reward = self._reward_func(actual_p_test, actual_p_quara) # shape: (env_count, zone_num)

        info = {}
        done = False
        if self.day > self.period:
            # 计算指标
            ep_r = self.history_cost['reward'].sum(dim=1).mean(dim=-1)  # (env_count,)
            total_infections = self.daily_new_I.sum(dim=[1,2])  # (env_count)
            total_test_num = self.history_cost['test_num'].sum(dim=[1,2]) # (env_count,)
            total_quara_num = self.history_cost['quara_num'].sum(dim=[1,2]) # (env_count,)
            total_quara_days = self.history_cost['quara_days'].sum(dim=[1,2]) # (env_count,)
            total_control_cost = total_quara_num + total_test_num * self.cost_ratio_t2q / (1/self.sigma + 1/self.gamma)
            score = (torch.exp((20 * 500 * total_infections) / self.TOTAL_POP)
                     + torch.exp(1 * 500 * total_control_cost / self.TOTAL_POP)) # (env_count)
            score_avg = (torch.exp((20 * 500 * total_infections.mean()) / self.TOTAL_POP[0])
                     + torch.exp(1 * 500 * total_control_cost.mean() / self.TOTAL_POP[0])) # (1)
            # rw = self.reward_weights.copy()
            # rw = [w / 100 for w in rw]
            # score = rw[0] * total_infections + rw[1] * total_test_num + rw[2] * total_quara_days
            # score = (torch.exp((rw[0] * 10 * total_infections) / self.TOTAL_POP)
            #          + torch.exp((rw[2] * 10 * total_control_cost) / self.TOTAL_POP))  # (env_count)
            # 增加一个 avg zone score 指标
            zone_infections  = self.daily_new_I.sum(dim=1)  # (env_count, zone_num)
            zone_test_num = self.history_cost['test_num'].sum(dim=1)    # (env_count, zone_num)
            zone_quara_num = self.history_cost['quara_num'].sum(dim=1)  # (env_count, zone_num)
            zone_control_cost = zone_quara_num + zone_test_num * self.cost_ratio_t2q / (1/self.sigma + 1/self.gamma)
            avg_zone_score = (torch.exp((20 * 500 * zone_infections) / self.POP)
                     + torch.exp(1 * 500 * zone_control_cost / self.POP)).mean(dim=-1)  # (env_count)
            print(f"({'Eval' if self.is_evaluation else 'Train'} Step End)\tTime: {datetime.now():%H:%M:%S}",
                  "\t reward:", f"{ep_r.mean().item():.4f}",
                  "\t total_infections:", f"{total_infections.mean().item():.4f}",
                  "\t total_test_num:", f"{total_test_num.mean().item():.4f}",
                  "\t total_quarantine_num:", f"{total_quara_num.mean().item():.4f}"
                  "\t total_quarantine_days:", f"{total_quara_days.mean().item():.4f}",
                  "\t total_control_cost:", f"{total_control_cost.mean().item():.4f}",
                  "\t score:", f"{score.mean().item():.4f}",
                  "\t avg_zone_score:", f"{avg_zone_score.mean().item():.4f}")
            # print('******\n', 'local_infe_cost:', self.history_cost['local_infe_cost'].sum(dim=1).mean(),
            #       'local_test_cost:', self.history_cost['local_test_cost'].sum(dim=1).mean(),
            #       'local_quara_cost:', self.history_cost['local_quara_cost'].sum(dim=1).mean())
            # print('global_infe_cost:', self.history_cost['global_infe_cost'].sum(dim=1).mean(),
            #       'global_test_cost:', self.history_cost['global_test_cost'].sum(dim=1).mean(),
            #       'global_quara_cost:', self.history_cost['global_quara_cost'].sum(dim=1).mean())
            if self.use_reward_shaping:
                print("penalty_times:", self.penalty_times)

            info = {
                'ep_r': ep_r,
                'total_infections': total_infections,
                'total_test_num': total_test_num,
                'total_quara_num': total_quara_num,
                'total_control_cost': total_control_cost,
                'total_quarantine_days': total_quara_days,
                'score': score,
                'score_avg': score_avg,
                'avg_zone_score': avg_zone_score,
            }

            done = True

        # 理论上done也要批次化，但对于这个环境，大家都是一起开始，一起结束的，所以done只是一个标量
        return obs, reward, done, info

    def _cal_detection_scale(self):
        """根据感染比例，确定检测到一个感染者所需要的人数"""
        EI_rate = self.simState[:, :, [self.E_undetected, self.I_undetected]].sum(dim=-1) / self.POP # (env_count, zone_num)
        scale = torch.zeros_like(EI_rate)
        mask = EI_rate != 0
        scale[mask] = 1 / torch.pow(EI_rate[mask], self.detection_efficiency_exp_param)
        scale[~mask] = 1e8 # 感染率为0时，所需检测人数填充为一个极大值
        return scale

        # f_min = 1.0
        # f_max = 1000.0
        #
        # return f_min + (f_max - f_min) * (1 - EI_rate)


    def _random_mask(self):
        # # 方式1：指定区域mask
        # mask = torch.zeros((self.env_count, self.ZONE_NUM), dtype=torch.bool, device=self.device)
        # mask[:, 0:int(self.ZONE_NUM * 0.15)] = True
        # return mask

        # 方式2：随机mask
        mask_rate_down = 0
        mask_rate_up = 0.3
        mask_duration_down = 14
        mask_duration_up = 28
        if self.R0 == 'low':
            mask_rate_down = 0.3
            mask_rate_up = 0.5
            mask_duration_down = 28
            mask_duration_up = 42
        mask_rate_down = self.mask_rate_down if self.mask_rate_down is not None else mask_rate_down
        mask_rate_up = self.mask_rate_up if self.mask_rate_up is not None else mask_rate_up
        mask_duration_down = self.mask_duration_down if self.mask_duration_down is not None else mask_duration_down
        mask_duration_up = self.mask_duration_up if self.mask_duration_up is not None else mask_duration_up

        rate = torch.rand(self.env_count, device=self.device) * (mask_rate_up - mask_rate_down) + mask_rate_down # 0.0-0.3之间的随机数
        mask = torch.rand(self.env_count, self.ZONE_NUM, device=self.device) < rate.unsqueeze(-1).expand(-1, self.ZONE_NUM)
        # # 对角线元素过大的区域mask掉的影响较大
        # mask[:, 563] = False
        cds = torch.randint(mask_duration_down, mask_duration_up, (self.env_count, 1), device=self.device).squeeze(-1)
        if not hasattr(self, 'old_zone_mask'):
            self.old_zone_mask = mask
            self.count_downs = cds
        else:
            # =0的环境重新计时
            cd_mask = self.count_downs <= 0
            self.count_downs[cd_mask] = cds[cd_mask]
            cd_mask = cd_mask.unsqueeze(-1).expand(-1, self.ZONE_NUM)
            self.old_zone_mask[cd_mask] = mask[cd_mask]
        self.count_downs -= 1
        return self.old_zone_mask

        # # 方式3：读取mask
        # if not hasattr(self, 'mask') or self.mask is None:
        #     raise NotImplementedError("mask not set")
        # return self.mask

    def set_mask(self, mask):
        self.mask = mask

    def _action_to_u(self, action):
        """将action（离散值）映射到两种控制措施上

        :param action: (env_count, ZONE_NUM)
                两种控制措施都是越大，越严格
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        # action = action.to(torch.long)
        u1_cnt = self.action_to_u1.shape[0]
        u_p_test = self.action_to_u0[(action / u1_cnt).to(torch.long)] * 0.1 # (env_count, ZONE_NUM)
        u_p_quara = self.action_to_u1[(action % u1_cnt).to(torch.long)] * 0.01 # (env_count, ZONE_NUM)
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

    def _get_obs(self):
        """获取观测值

        :return: (env_count, ZONE_NUM, window_size, local_obs_dim * 2)
        """
        """观测1：对现存EI的观测和估计，对新增E的估计"""
        # curr known EI
        curr_known_EI = self.simState[:, :, [self.E_detected, self.I_reported, self.I_detected]].sum(dim=-1)
        estimate_Eun = self.daily_detect_E[:, self.day - 1, :] / self.detect_E_rate * (1 - self.detect_E_rate)  # 代表 Eun 仓室，估计准确的前提是检测率test_num_rate达到1
        estimate_Iun =  self.daily_detect_I[:, self.day - 1, :] / self.detect_I_rate * (1 - self.detect_I_rate) # 代表 Iun 仓室
        # new E
        # estimate_new_E = self.daily_detect_E[:, self.day - 1, :] + self.daily_detect_I[:, self.day - 1, :]    # 用新检测出的E，表示对新增E的估计
        estimate_new_E = self.daily_detect_E[:, self.day - 1, :] + self.daily_detect_I[:, self.day - 1, :] + self.daily_new_report[:, self.day - 1, :]    # 用新观测出的EI，表示对新增E的估计
        # estimate_new_E = self.daily_detect_E[:, self.day - 1, :]    # 用新检测出的E，表示对新增E的估计

        curr_obs = [curr_known_EI + estimate_Eun + estimate_Iun, estimate_new_E] # [现存的，Eun]

        # curr_known_EI += self.simState[:, :, [self.QE, self.QI]].sum(dim=-1)

        # """观测2：对现存I的观测和估计，对新增I的估计"""
        # # curr known I
        # curr_known_I = self.simState[:, :, [self.I_reported, self.I_detected]].sum(dim=-1)
        # estimate_Iun = self.daily_detect_I[:, self.day - 1, :] / self.detect_I_rate * (1 - self.detect_I_rate) # 表示对当前Iun的估计
        # # new I 的估计
        if self.state_contain_detected:
            I_detected = self.simState[:, :, self.I_detected]
            E_detected = self.simState[:, :, self.E_detected]
            curr_obs.append(I_detected)
            curr_obs.append(E_detected)
        if self.state_contain_Q:
            Q = self.simState[:, :, [self.QE, self.QI]].sum(dim=-1)
            curr_obs.append(Q)
        if self.state_contain_R:
            total_knownR = self.daily_known2R.sum(dim=1) # shape: (env_count, zone_num)
            curr_obs.append(total_knownR)
        # # 归一化
        curr_obs = torch.stack(curr_obs, dim=-1)
        curr_obs = curr_obs / self.POP.unsqueeze(-1)
        if self.state_contain_detect_rate:
            action = self.actions[:, self.day - 2, :]
            p_test, _ = self._action_to_u(action)  # (env_count, ZONE_NUM)
            curr_obs = torch.cat([curr_obs, p_test.unsqueeze(-1)], dim=-1)
            # # curr action
            # action = self.actions[:, self.day - 2, :].unsqueeze(-1).clone() # shape: (env_count, zone_num)
            # curr_obs = torch.cat([curr_obs, action], dim=-1)

        self.history_local_obs[:, self.day - 1, :, :] = curr_obs # shape: (env_count, zone_num, local_obs_dim)

        local_obs = self._get_history_padding(self.history_local_obs, self.day, self.WINDOW_SIZE) # shape: (env_count, WINDOW_SIZE, ZONE_NUM, local_obs_dim)
        global_obs = local_obs.mean(dim=2) # shape: (env_count, WINDOW_SIZE, local_obs_dim)
        global_obs_expanded = global_obs.unsqueeze(2).repeat(1, 1, local_obs.size(2), 1)
        combined_obs = torch.cat([local_obs, global_obs_expanded], dim=-1) # shape: (env_count, WINDOW_SIZE, ZONE_NUM, local_obs_dim * 2)

        return combined_obs

    def _reward_func(self, actual_p_test, actual_p_quara):
        """计算每个区域得到的奖励

        说明：每个区域都有自己的时间秩序度，但空间秩序度只出现在全局奖励上

        :return shape:(env_count, zone_num)
        """

        # 1.感染成本（E， I， new）
        # # a.按各区域的现存I计算
        # infections = state[:, :, [self.I_undetected, self.I_detected, self.I_reported]].sum(dim=-1) # (env_count, zone_num)
        # local_infe_cost = (20 * infections) ** 2 / self.POP # (env_count, zone_num)
        # global_infe_cost = (20 * infections.sum(dim=1)) ** 2 / self.TOTAL_POP # (env_count,)

        # # 按各区域新增的I计算
        # new_infections = self.daily_new_E[:, self.day - 1, :] + self.daily_new_I[:, self.day - 1, :] # (env_count, zone_num)
        # avg_infections = self.daily_new_I.sum(dim=1) / (self.day - 1)  # (env_count, zone_num)
        # # new_infections = avg_infections
        # local_infe_cost = (20 * new_infections) ** 2 / self.POP  # (env_count, zone_num)
        # global_infe_cost = (20 * new_infections.sum(dim=1)) ** 2 / self.TOTAL_POP  # (env_count,)

        if not self.disable_response_distribution:
            # 按各区域导致的新增E计算
            new_E_self_cause = self.daily_new_E_self_cause[:, self.day - 1, :] # (env_count, zone_num)
            local_infe_cost = new_E_self_cause / self.POP # 即时奖励
            global_infe_cost = (new_E_self_cause.sum(dim=1)) / self.TOTAL_POP # 全局奖励
        else:
            # 按各区域自己的新增E计算
            new_E_self_region = self.daily_new_E[:, self.day - 1, :]
            local_infe_cost = new_E_self_region / self.POP
            global_infe_cost = (new_E_self_region.sum(dim=1)) / self.TOTAL_POP

        # 2.检测成本 actual_p_test
        test_num = actual_p_test * self.POP # (env_count, zone_num)
        self.daily_test_num[:, self.day - 1, :] = test_num
        local_test_cost = test_num / self.POP # (env_count, zone_num)
        global_test_cost = test_num.sum(dim=1) / self.TOTAL_POP # (env_count,)

        # 3.隔离成本 Q
        quara_num = self.daily_new_Q[:, self.day - 1, :] # shape: (env_count, zone_num)
        self.daily_quara_num[:, self.day - 1, :] = quara_num
        # quara_num取值quara_num和actual_p_quara较大的
        if not self.is_evaluation:
            quara_num = torch.max(quara_num, actual_p_quara * 20)
            # quara_num = torch.max(quara_num, actual_p_quara * 200)
        local_quara_cost = quara_num / self.POP # (env_count, zone_num)
        global_quara_cost = quara_num.sum(dim=-1) / self.TOTAL_POP # (env_count,)

        rw = self.reward_weights.copy()

        # 1.global reward
        global_reward = - (rw[0] * global_infe_cost + rw[1] * global_test_cost + rw[2] * global_quara_cost) # shape: (env_count,)
        global_reward_expaned = global_reward.unsqueeze(1).repeat(1, self.ZONE_NUM) # shape: (env_count, zone_num)

        # 2.local reward
        local_reward = -(rw[0] * local_infe_cost + rw[1] * local_test_cost + rw[2] * local_quara_cost) # shape: (env_count, zone_num)
        reward = self.local_reward_weight * local_reward + (1 - self.local_reward_weight) * global_reward_expaned # shape: (env_count, zone_num)
        # if self.day in [10,20,30,40,50,60,70,80,90,100,110]:
        #     print(f'day:{self.day}-------------------')
        #     print('local_infe_cost:', local_infe_cost[:,:5].mean(dim=0))
        #     print('local_test_cost:', local_test_cost[:,:5].mean(dim=0))
        #     print('local_quara_cost:', local_quara_cost[:,:5].mean(dim=0))
        #     print('reward:', reward[:,:5].mean(dim=0))


        if self.use_reward_shaping:
            # 如果有上报人数，但该区域没有检测，就给予惩罚，上报人数应该是前一天的
            state = self.simRes[:, self.day - 2, :, :]
            infections = state[:, :, [self.E_detected, self.I_reported, self.I_detected]].sum(dim=-1) # (env_count, zone_num)
            infections_large = infections > 0.1
            action_small = actual_p_test == 0 # (env_count, zone_num)
            zone_mask = infections_large & action_small
            # 如果感染人数没有，但采取较强的控制措施
            infections_small = infections < 0.0005
            action_large = actual_p_test >= 0.005
            # zone_mask = (infections_small & action_large)
            zone_mask = (infections_small & action_large) | zone_mask
            if zone_mask.any():
                self.penalty_times += zone_mask.sum().item()
            # 施加惩罚，例如增加一个大的负值
            penalty_factor = 2  # 惩罚因子
            penalty = penalty_factor * (zone_mask.float())
            reward -= penalty

        # 记录奖励，确保为每个批次独立记录
        self.history_cost['reward'][:, self.day - 2, :] = reward.clone()
        self.history_cost['test_num'][:, self.day - 2, :] = test_num
        self.history_cost['quara_num'][:, self.day - 2, :] = quara_num
        self.history_cost['quara_days'][:, self.day - 2, :] = self.simState[:, :, [self.QE,self.QI]].sum(dim=2)
        self.history_cost['local_reward'][:, self.day - 2, :] = local_reward
        self.history_cost['global_reward'][:, self.day - 2] = global_reward
        self.history_cost['local_infe_cost'][:, self.day - 2] = local_infe_cost.mean(dim=-1)
        self.history_cost['local_test_cost'][:, self.day - 2] = local_test_cost.mean(dim=-1)
        self.history_cost['local_quara_cost'][:, self.day - 2] = local_quara_cost.mean(dim=-1)
        self.history_cost['global_infe_cost'][:, self.day - 2] = global_infe_cost.mean(dim=-1)
        self.history_cost['global_test_cost'][:, self.day - 2] = global_test_cost.mean(dim=-1)
        self.history_cost['global_quara_cost'][:, self.day - 2] = global_quara_cost.mean(dim=-1)


        return reward

    def __delay_quara(self, quara_plan_num):
        """将今日的计划隔离人数延迟1-2天
        :param quara_plan_num: (env_count, zone_num)
        """
        if self.use_delay_quara:
            # 分散到 今天，明天，后天
            for i, r in enumerate([0.3, 0.4, 0.3]):
                if self.day - 1 + i > self.period:
                    continue
                self.daily_delay_quara_num[:, self.day - 1 + i, :] += r * quara_plan_num
        else:
            self.daily_delay_quara_num[:, self.day - 1, :] += quara_plan_num

    def __cal_pearsonr(self):
        mask = torch.rand(self.ZONE_NUM) < 0.3  # 小于比例的为 True
        curr_EI = self.simState[0, :, [self.E_undetected, self.E_detected,
                                       self.I_undetected, self.I_detected, self.I_reported]].sum(dim=-1)
        # curr_EI = curr_EI / self.POP[0]
        know_EI = torch.zeros_like(curr_EI)
        know_EI[~mask] = curr_EI[~mask]

        OD = self.OD[0]
        OD_t = OD.transpose(0, 1)

        pre = torch.mm(OD_t, know_EI.unsqueeze(1)) / torch.mm(OD_t, self.POP[0].unsqueeze(1))
        pre = torch.mm(OD, pre)[:, 0] * self.POP[0]

        # 数值修正
        pre = pre * (curr_EI[~mask].sum() / pre[~mask].sum() + 1e-12)

        y = curr_EI[mask]
        x = pre[mask]

        # y = (curr_EI / self.POP[0])[mask]
        # x = (pre[:, 0] / self.POP[0])[mask]

        from scipy.stats import pearsonr
        r, p_value = pearsonr(x.cpu().numpy(), y.cpu().numpy())
        print(f"Pearson Correlation Coefficient: {r:.4f}")
        print(f"P-value: {p_value:.4e}")

        # import pandas as pd
        # data = pd.DataFrame({'x': x.numpy(), 'y': y.numpy()})
        # data.to_csv('output.csv', index=False)

    def extract_observe(self):
        obs = self.history_local_obs[:, :, :, :2] * self.POP.unsqueeze(1).unsqueeze(-1)
        return obs
    def extract_daily_quara_num(self):
        daily_quara_num = self.daily_quara_num.clone()
        return daily_quara_num

    def record_rebuild_state(self, rebuild_state: torch.Tensor):
        assert rebuild_state.shape == (self.env_count, self.ZONE_NUM, 2), "rebuild_state shape error. expected (env_count, zone_num, 2)"
        self.rebuild_states.append(rebuild_state)
    def extract_rebuild_state(self):
        if len(self.rebuild_states) == self.period:
            print("env.rebuild_state length: ", len(self.rebuild_states))
        if len(self.rebuild_states) == 0:
            return torch.zeros(1,1,1,0)
        r_s = torch.stack(self.rebuild_states, dim=1)
        return r_s

    def get_observation_missing_rate(self):
        """
        获取自上次 reset 以来的观测缺失率。

        返回：各个环境的观测缺失率，形状 (env_count,)
        """
        total_zone_days = self.ZONE_NUM * (self.day - 1)
        missing_rate = self.obs_missing_total_days.sum(dim=-1) / total_zone_days
        return missing_rate

    def render_all_rooms(self):
        # 绘制所有仓室
        S = self.simRes[:, :, :, self.S].sum(dim=2).mean(dim=0) # shape: (period+1)
        E_undetected = self.simRes[:, :, :, self.E_undetected].sum(dim=2).mean(dim=0)
        E_detected = self.simRes[:, :, :, self.E_detected].sum(dim=2).mean(dim=0)
        I_undetected = self.simRes[:, :, :, self.I_undetected].sum(dim=2).mean(dim=0)
        I_detected = self.simRes[:, :, :, self.I_detected].sum(dim=2).mean(dim=0)
        I_reported = self.simRes[:, :, :, self.I_reported].sum(dim=2).mean(dim=0)
        QE = self.simRes[:, :, :, self.QE].sum(dim=2).mean(dim=0)
        QI = self.simRes[:, :, :, self.QI].sum(dim=2).mean(dim=0)
        R = self.simRes[:, :, :, self.R].sum(dim=2).mean(dim=0)

        # 累计感染者变化
        daily_new_I = self.daily_new_I.sum(dim=2).mean(dim=0)
        daily_new_I_cum = daily_new_I.cumsum(dim=0) # shape: (period+1)

        # 每日新增


        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        plt.plot(S.cpu(), label='S (avg)', linewidth=1)
        plt.plot(E_undetected.cpu(), label='E_undetected (avg)', linewidth=1)
        plt.plot(E_detected.cpu(), label='E_detected (avg)', linewidth=1)
        plt.plot(I_undetected.cpu(), label='I_undetected (avg)', linewidth=1)
        plt.plot(I_detected.cpu(), label='I_detected (avg)', linewidth=1)
        plt.plot(I_reported.cpu(), label='I_reported (avg)', linewidth=1)
        plt.plot(QE.cpu(), label='QE (avg)', linewidth=1)
        plt.plot(QI.cpu(), label='QI (avg)', linewidth=1)
        plt.plot(R.cpu(), label='R (avg)', linewidth=1)

        plt.plot(daily_new_I_cum.cpu(), label='daily_new_I_cum (avg)', linewidth=2, color='red', linestyle='--')
        plt.plot(daily_new_I.cpu(), label='daily_new_I (avg)', linewidth=2, color='red')

        plt.legend(fontsize=10, loc='upper right')
        plt.xlim(0, self.period)
        plt.show()

    def render(self, title='', fig_dir=None) :
        plt.figure(dpi=120, figsize=(7, 5))
        plt.grid(linestyle='-.', axis='both')

        # 计算所有环境的感染总和并取平均
        # infections = self.simRes[:, :, :, [self.I_undetected, self.I_detected, self.I_reported]].sum(dim=[2,3]).mean(dim=0)  # (period+1,)
        # plt.plot(infections.cpu(), label='Infections (avg)', color='orange', linewidth=2)
        # exposeds = self.simRes[:, :, :, [self.E_undetected, self.E_detected]].sum(dim=[2,3]).mean(dim=0)
        # plt.plot(exposeds.cpu(), label='Exposeds (avg)', color='green', linewidth=2)

        # 观测到的EI
        estimate_total_EI = (self.history_local_obs[:, :, :, 0] * self.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)
        plt.plot(estimate_total_EI.cpu(), label='Obs Total EI (avg)', color='red', linewidth=2, linestyle='--', alpha=0.5)
        # 观测的新增EI
        estimate_new_E = (self.history_local_obs[:, :, :, 1] * self.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)
        # plt.plot(estimate_new_E.cpu(), label='Obs New E (avg)', color='green', linewidth=2, linestyle='--', alpha=0.5)

        # 总的EI
        total_EI = self.simRes[:, :, :, [self.E_undetected, self.E_detected,
                                         self.I_undetected, self.I_detected, self.I_reported]].sum(dim=[2,3]).mean(dim=0)
        plt.plot(total_EI.cpu(), label='Actual Total EI (avg)', color='red', linewidth=2, alpha=0.5)
        # 实际的新增EI
        actual_new_E = self.simRes[:, :, :, self.E_undetected].sum(dim=2).mean(dim=0)
        # plt.plot(actual_new_E.cpu(), label='Actual New E (avg)', color='green', linewidth=2, alpha=0.5)


        # # E_undetected的变化（观测外来输入）
        # E_undetected = self.simRes[:, :, :, [self.E_undetected]].sum(dim=[2,3]).mean(dim=0)
        # plt.plot(E_undetected.cpu(), label='E_undetected (avg)', color='green', linewidth=2)

        plt.axhline(ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0)

        plt.legend(fontsize=12, loc='upper left')
        plt.twinx()

        # 绘制总的检测人数，总的感染人数
        reward = self.history_cost['reward'].mean(dim=0)
        test_num = self.history_cost['test_num'].mean(dim=0)
        quara_days = self.history_cost['quara_days'].mean(dim=0)
        local_reward = self.history_cost['local_reward'].mean(dim=0)
        global_reward = self.history_cost['global_reward'].mean(dim=0)

        # plt.plot(test_num.cpu(), label='Test_num (avg)', color='green', linewidth=1)
        # plt.plot(quara_num.cpu(), label='Quara_num (avg)', color='red', linewidth=1)

        # 绘制两类动作的变化曲线
        action = self.actions
        u0, u1 = self._action_to_u(action)
        u0, u1 = u0.mean(dim=[0,2]), u1.mean(dim=[0,2]) # (period+1)
        plt.plot(u0.cpu(), label='Test rate (avg)', linewidth=1, linestyle='--', color='green', alpha=0.5)
        plt.plot(u1.cpu(), label='Quara rate (avg)', linewidth=1, linestyle='--', color='green')
        # plt.ylim(0, 0.1)
        plt.ylim(0, )

        plt.legend(fontsize=12, loc='upper right')

        plt.title("Daily Current Infection (avg) (" + self.city + ")" if title == '' else title)

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

        # 方法需要重写

        plt.show()

    def render_one_region(self, title='', env_idx=0, region_idx=0) :
        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        # 观测到的EI
        estimate_total_EI = (self.history_local_obs[env_idx, :, region_idx, 0] * self.POP[env_idx, region_idx])
        plt.plot(estimate_total_EI.cpu(), label='estimate_total_EI', color='red', linewidth=2, linestyle='--',
                 alpha=0.5)
        # 观测的新增EI
        estimate_new_E = (self.history_local_obs[env_idx, :, region_idx, 1] * self.POP[env_idx, region_idx])
        plt.plot(estimate_new_E.cpu(), label='estimate_new_E', color='green', linewidth=2, linestyle='--',
                 alpha=0.5)

        # 总的EI
        total_EI = self.simRes[env_idx, :, region_idx, [self.E_undetected, self.E_detected,
                                         self.I_undetected, self.I_detected, self.I_reported]].sum(dim=-1)
        plt.plot(total_EI.cpu(), label='Total EI', color='red', linewidth=2, alpha=0.5)
        # 实际的新增EI
        # actual_new_E = self.daily_new_E[env_idx, :, region_idx]
        actual_new_E = self.simRes[env_idx, :, region_idx, self.E_undetected]
        plt.plot(actual_new_E.cpu(), label='Actual New E', color='green', linewidth=2, alpha=0.5)

        plt.axhline(ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, )

        plt.legend(fontsize=8, loc='upper left')
        plt.twinx()

        # 绘制两类动作的变化曲线
        action = self.actions[env_idx, :, region_idx] # [period+1]
        u0, u1 = self._action_to_u(action)
        plt.plot(u0.cpu(), label='Action_0 (avg)', linewidth=1, linestyle='--')
        plt.plot(u1.cpu(), label='Action_1 (avg)', linewidth=1, linestyle='--')
        # plt.ylim(0, action_to_u.size(0))

        plt.legend(fontsize=8, loc='upper right')

        plt.title(f"Daily Current Infection (env idx={env_idx}, region idx={region_idx}) (" + self.city + ")" if title == '' else title)
        plt.show()

    def close(self):
        print('close Environment')
        pass

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
    # args.use_obs_imperfect = True
    # args.obs_imperfect_down = 0.5
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
    # args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.5

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

def calculate_R0_from_params(env):
    # 潜伏者贡献
    e_contribution = env.beta * env.beta_E_rate * (1 / env.sigma)
    # 感染者贡献
    i_contribution = env.beta * env.Pm * (1 / env.gamma)
    # 总R0
    R0 = e_contribution + i_contribution
    return R0
def exponential_growth(t, r, C0):
    """指数增长模型：N(t) = C0 * exp(r*t)"""
    return C0 * np.exp(r * t)

def calculate_R0_from_simulation(env, start=1, end=30):
    from scipy.optimize import curve_fit
    # 1. 提取模拟数据（每日新增感染数）
    daily_new_I = env.daily_new_I.mean(dim=0).sum(dim=-1).cpu().numpy()  # 汇总所有环境和区域
    # 取指数增长阶段（前30天，假设此时易感人群充足）
    t = np.arange(start, end)
    new_cases = daily_new_I[start:end]  # 跳过第0天

    # 2. 拟合指数增长模型，求增长率r
    try:
        popt, _ = curve_fit(exponential_growth, t, new_cases, p0=(0.1, 1))
        r = popt[0]  # 增长率
    except:
        return np.nan

    # 3. 计算代际间隔 T_g（潜伏期+传染期的平均）
    T_g = (1/env.sigma.item() + 1/env.gamma.item()) / 2  # 单位：天

    # 4. 计算R0
    R0 = 1 + r * T_g + (r * T_g) ** 2
    return R0
def robust_R0_calculation(env):
    R0_list = []
    for start in range(1, 16):
        R0 = calculate_R0_from_simulation(env, start=start, end=30)
        print(f"start: {start}, R0: {R0:.4f}")
        R0_list.append(R0)
    print(f"R0: {min(R0_list)} - {max(R0_list)}")
def cal_R0():
    from config import args

    args.simulate_scale = 'community'
    args.zone_num = 654

    # 低R0实验
    args.ODE_beta = 0.4
    args.R0 = 'low'

    args.state_contain_detected = True
    args.state_contain_Q = False
    args.state_contain_R = False
    args.state_contain_detect_rate = False
    args.local_obs_dim = 2 + 2 * args.state_contain_detected + args.state_contain_Q + args.state_contain_R + args.state_contain_detect_rate

    # args.device_name = 'cpu'
    device = torch.device(args.device_name)

    env = EpidemicModel(args, env_count=20)
    env.seed(3047)
    env.reset()

    while True:
        action = torch.zeros(env.env_count, env.ZONE_NUM).to(device)

        s_, r, done, info = env.step(action=action)

        if done:
            env.render()
            env.render_all_rooms()
            # env.render_one_region()
            # 总感染的峰值天数需要对每个批次进行计算
            daily_new_I = env.daily_new_I.sum(dim=-1).cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, [1, 2, 3, 4, 5]].sum(dim=(2, 3))
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"总新增感染人数：{total_new_I}，现存感染最大峰值：{daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")

            break

    # 计算R0
    R0_param = calculate_R0_from_params(env)  # 基于参数
    R0_simulation = calculate_R0_from_simulation(env)  # 基于模拟数据
    robust_R0_calculation(env)
    print(f"基于参数的R0: {R0_param:.2f}")
    print(f"基于模拟的R0: {R0_simulation:.2f}")

def natural_transmission():
    from config import args
    args.simulate_scale = 'community'
    args.zone_num = 654

    # args.device_name = 'cpu'
    device = torch.device(args.device_name)

    env = EpidemicModel(args, env_count=1)
    env.seed(3047)
    env.reset()

    while True:
        action = torch.zeros(env.env_count, env.ZONE_NUM).to(device)

        s_, r, done, info = env.step(action=action)

        if done:
            env.render()
            env.render_all_rooms()
            # 总感染的峰值天数需要对每个批次进行计算
            daily_new_I = env.daily_new_I.sum(dim=-1).cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, [1, 2, 3, 4, 5]].sum(dim=(2, 3))
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"总新增感染人数：{total_new_I}，现存感染最大峰值：{daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")

            break

    # 将env的simRes导出到 ./draw/模拟器可视化/simRes.npy
    np.save('../draw/模拟器可视化/simRes.npy', env.simRes.cpu().numpy())

def output_imperfect_observation(output_dir="../draw/分析自然传播下三种观测等级的差异/data"):
    from config import args
    args.simulate_scale = 'community'
    args.zone_num = 654

    # args.ODE_beta = 0.4
    # args.R0 = 'low'

    # args.device_name = 'cpu'
    device = torch.device(args.device_name)

    env = EpidemicModel(args, env_count=100)
    env.seed(3047)
    env.reset()

    while True:
        action = torch.zeros(env.env_count, env.ZONE_NUM).to(device)
        s_, r, done, info = env.step(action=action)

        if done:
            env.render()
            env.render_all_rooms()
            # 输出真实状态和完备观测数据
            simRes = env.simRes.cpu().numpy()
            perfect_obs = env.extract_observe().cpu().numpy()
            daily_new_E = env.daily_new_E.cpu().numpy()
            # np.save(f"{output_dir}/simRes_low.npy", simRes)
            np.save(f"{output_dir}/simRes1.npy", simRes)
            np.save(f"{output_dir}/perfect_obs.npy", perfect_obs)
            np.save(f"{output_dir}/daily_new_E.npy", daily_new_E)
            break

    # 部分可观测
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1
    env = EpidemicModel(args, env_count=100)
    env.seed(3047)
    env.reset()

    while True:
        action = torch.zeros(env.env_count, env.ZONE_NUM).to(device)
        s_, r, done, info = env.step(action=action)

        if done:
            env.render()
            env.render_all_rooms()
            # 输出真实状态和部分可观测数据
            simRes = env.simRes.cpu().numpy()
            partial_obs = env.extract_observe().cpu().numpy()
            np.save(f"{output_dir}/simRes2.npy", simRes)
            np.save(f"{output_dir}/partial_obs.npy", partial_obs)
            print("missing rate:", env.get_observation_missing_rate().mean())
            break


if __name__ == '__main__':
    # measure_run_time()
    # exit(0)
    # test_control_timely()
    # exit(0)
    # cal_R0()
    # exit(0)
    # natural_transmission()
    # exit(0)
    output_imperfect_observation()
    exit(0)
    from config import args


    args.simulate_scale = 'community'
    args.zone_num = 654

    args.state_contain_detected = True
    args.state_contain_Q = False
    args.state_contain_R = False
    args.state_contain_detect_rate = False
    args.local_obs_dim = 2 + 2 * args.state_contain_detected + args.state_contain_Q + args.state_contain_R + args.state_contain_detect_rate

    # 不完全观测
    # args.use_obs_imperfect = True
    args.obs_imperfect_down = 1.0

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

    args.ODE_detect_E_rate = 0
    # args.ODE_detect_I_rate = 1.0

    env = EpidemicModel(args, env_count=2)
    env.seed(3047)
    env.reset()

    # 动态变化的beta
    if env.use_beta_change:
        env.dynamic_beta_change(type=args.env_beta_change_rule)


    while True:

        if env.day == 45 and env.gather_to_some_region:
            env.adjust_OD(regions=regions, increase_ratio=increase_ratio, beta_increase_ratio=beta_increase_ratio)

        action=torch.ones(env.env_count, env.ZONE_NUM).to(device)

        action = action * 4

        s_, r, done, info = env.step(action=action)

        if env.day == 50:
            env.reset_OD()

        if done:
            # (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
            #  ep_infections_rate, ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()
            #
            env.render()
            env.render_all_rooms()
            # env.render_one_region()
            # 总感染的峰值天数需要对每个批次进行计算
            daily_new_I = env.daily_new_I.sum(dim=-1).cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, [1,2,3,4,5]].sum(dim=(2,3))
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"总新增感染人数：{total_new_I}，现存感染最大峰值：{daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")

            break