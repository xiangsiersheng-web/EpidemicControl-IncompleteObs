# -*- encoding: utf-8 -*-
# @Time    : 2024-08-30
# @File    : meta_env_tensor.py

import random

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from dynamic.env_change_rules.beta_change import BetaChangeRule


class EpidemicModelTensor:
    def __init__(self, args):
        args_dict = vars(args)
        self.reward_mode = args.reward_mode
        self.city = args.city
        self.R0 = args.R0
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device_name)

        data_dir = './data/' + self.city if 'data_dir' not in args_dict else args.data_dir
        # 子区域邻接关系
        self.adj_matrix = torch.from_numpy(np.load(data_dir + '/adj_matrix.npy')).float().to(self.device)
        self.flow_top3_matrix = torch.from_numpy(np.load(data_dir + '/flow_top3_matrix.npy')).float().to(self.device)
        if self.city == 'sz':
            self.adm_matrix = torch.from_numpy(np.load(data_dir + '/adm_matrix.npy')).float().to(self.device)
        self.action_max = 3 # 注意，action的最大值是写死了的

        # 子区域间流动
        self.OD = torch.from_numpy(np.load(data_dir + '/flow.npy')).float().to(self.device)

        # 子区域人口数量
        self.POP = torch.from_numpy(np.load(data_dir + '/population.npy')).float().to(self.device)
        self.ZONE_NUM = self.POP.shape[0]

        # 传染病相关参数（感染者移动比例、beta、潜伏期、恢复期）
        self.Pm = torch.tensor(0.4 if "ODE_Pm" not in args_dict else args.ODE_Pm, device=self.device)
        self.beta = torch.tensor(0.8 if "ODE_beta" not in args_dict else args.ODE_beta, device=self.device)
        self.betas = torch.full((self.ZONE_NUM,), self.beta.item(), device=self.device)
        self.sigma = torch.tensor(1 / 3 if "ODE_sigma" not in args_dict else args.ODE_sigma, device=self.device)
        self.gamma = torch.tensor(1 / 7 if "ODE_gamma" not in args_dict else args.ODE_gamma, device=self.device)

        # 是否使用beta变化
        print("动态环境为：", args.beta_change_rule)
        self.use_beta_change = False
        if args.env_beta_change_rule != BetaChangeRule.NONE:
            self.use_beta_change = True
            self.beta_matrix = torch.from_numpy(args.beta_matrix).float().to(self.device)

        # 不同城市的容量和ylim（绘图用）
        city_capacity = {'sz': 4e6, 'tokyo': 2.2e6, 'nyc': 2e6, 'sh': 5.5e6}
        self.capacity = torch.tensor(city_capacity[self.city], device=self.device)
        self.capacity = self.capacity if self.R0 == 'high' else self.capacity / 2

        city_ylim = {'sz': 9e6,'tokyo': 6e6,'nyc': 6e6,'sh': 1.2e7}
        self.ylim = city_ylim[self.city]

        # 模拟周期
        self.period = 120 if "ODE_period" not in args_dict else args.ODE_period

        # 状态的观测天数
        self.WINDOW_SIZE = 7 if "WINDOW_SIZE" not in args_dict else args.WINDOW_SIZE

        self.reset()


    def adjust_env_params(self, **kwargs):
        """用于动态调用环境参数"""
        if 'beta' in kwargs and 'betas' in kwargs:
            raise ValueError("Cannot specify both 'beta' and 'betas'.")
        if 'beta' in kwargs:
            if (self.beta != kwargs['beta']):
                self.beta = kwargs['beta']
                self.betas = np.array([self.beta] * self.ZONE_NUM)
        if 'betas' in kwargs:
            self.betas = np.array(kwargs['betas'])
        if len(self.betas) != self.ZONE_NUM:
            raise ValueError("Length of 'betas' must be equal to ZONE_NUM.")
        # for k, v in kwargs.items():
        #     setattr(self, k, v) # 调整对象的属性


    def spatio_entropy_adj_matrix(self, action, adj_matrix):
        """
        使用优化的Tensor操作计算空间熵。
        :param action: Tensor, 各节点的动作值（整数1到3）
        :param adj_matrix: Tensor, 邻接矩阵
        :return: tensor.float, 计算得到的总空间熵
        """
        epsilon = 1e-9  # 防止除零和log(0)

        # 扩展action到每行都是完整的action复制
        extended_action = action.unsqueeze(0).repeat(action.size(0), 1)

        # 应用邻接矩阵，过滤邻接节点的行为
        filtered_actions = extended_action * adj_matrix

        # 将非邻接的部分设置为0
        filtered_actions[adj_matrix == 0] = -1  # 使用不在[0, action_max)范围内的数字标记非邻接

        # 计算每个动作的频率
        action_counts = torch.zeros(filtered_actions.size(0), self.action_max, device=filtered_actions.device)
        for i in range(0, self.action_max):
            action_counts[:, i] = (filtered_actions == i).sum(dim=1)

        # 计算概率
        row_sums = action_counts.sum(dim=1, keepdim=True)
        probabilities = action_counts / row_sums

        # 计算熵
        entropy = -(probabilities * torch.log2(probabilities + epsilon))
        row_entropy = entropy.sum(dim=1)
        total_entropy = row_entropy.sum()

        return total_entropy

    def spatio_entropy_adm(self, action, adm_matrix):
        """
        使用优化的Tensor操作计算空间熵，专用于街道
        :param action: Tensor, 各节点的动作值（整数1到3）
        :param adm_matrix: Tensor, 街道邻接
        :return: tensor.float, 计算得到的总空间熵
        """
        epsilon = 1e-9  # 防止除零和log(0)

        # 扩展action到每行都是完整的action复制
        extended_action = action.unsqueeze(0).repeat(adm_matrix.size(0), 1)

        # 应用邻接矩阵，过滤邻接节点的行为
        filtered_actions = extended_action * adm_matrix

        # 将非邻接的部分设置为0
        filtered_actions[adm_matrix == 0] = -1  # 使用不在[0, action_max)范围内的数字标记非邻接

        # 计算每个动作的频率
        action_counts = torch.zeros(filtered_actions.size(0), self.action_max, device=filtered_actions.device)
        for i in range(0, self.action_max):
            action_counts[:, i] = (filtered_actions == i).sum(dim=1)

        # 计算概率
        row_sums = action_counts.sum(dim=1, keepdim=True)
        probabilities = action_counts / row_sums

        # 计算熵
        entropy = -(probabilities * torch.log2(probabilities + epsilon))

        row_entropy = entropy.sum(dim=1)
        # adm 矩阵的行和作为权重
        w = adm_matrix.sum(dim=1)
        total_entropy = (row_entropy * w).sum()

        return total_entropy

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
        self.actions = torch.zeros((self.period + 1, self.ZONE_NUM), device=self.device) # 初始化动作数组为零向量，确保在正确的设备上

        # 初始化历史成本数据
        self.history_cost = {"reward": torch.zeros((self.period, ), device=self.device),
                             "sdo": torch.zeros((self.period, ), device=self.device),
                             "fdo": torch.zeros((self.period, ), device=self.device),
                             "ado": torch.zeros((self.period, ), device=self.device)}

        # 初始化仿真状态，将人口数据设置到每个子区域
        self.simState = torch.zeros((self.ZONE_NUM, 4), device=self.device).float()
        self.simState[:, 0] = self.POP

        # 设置初始感染者种子
        self.set_init_seed(rand_idxs=rand_idxs)

        # 初始化仿真结果数组
        self.simRes = torch.zeros((self.period + 1, self.ZONE_NUM, 4), device=self.device)
        self.simRes[0] = self.simState

        # 使用PyTorch进行填充操作，以替代np.pad
        if self.day >= self.WINDOW_SIZE:
            Is_window = self.simRes[(self.day - self.WINDOW_SIZE): self.day, :, :]
        else:
            padding = self.WINDOW_SIZE - self.day
            current_data = self.simRes[:self.day, :, 2]
            Is_window = torch.nn.functional.pad(current_data, (0, 0, padding, 0), "constant", 0)
        obs = Is_window.flatten()

        # 重置日常新发事件数组
        self.daily_new_E = torch.zeros((self.period, ), device=self.device)
        self.daily_new_I = torch.zeros((self.period, ), device=self.device)
        self.daily_attack_rate = torch.zeros((self.period, ), device=self.device)

        return obs

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
            rand_idxs = torch.tensor(rand_list).to(self.device)

        # # 使用PyTorch的索引功能来增加感染者人数
        # self.simState[rand_idxs, 1] += 1
        # self.simState[rand_idxs, 0] -= 1
        # 使用torch.bincount计算每个索引的出现次数
        counts = torch.bincount(rand_idxs, minlength=self.simState.shape[0]).float()

        # 更新self.simState中对应索引的第一、二列
        self.simState[:, 1] += counts
        self.simState[:, 0] -= counts

    def step(self, action):
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
        # action 保证是tensor，且在device上
        assert type(action) == torch.Tensor and action.device == self.device, "action must be a tensor on device"
        if self.use_beta_change:
            self.betas = self.beta_matrix[self.day - 1]
        self.day += 1
        self.actions[self.day - 1] = action
        # self.actions.append(action)  # TODO: 考虑如何处理此列表以便于性能优化

        # 仓室模型step
        contagious_infects = self.simState[:, 1] + self.Pm * self.simState[:, 2]

        contagious_toJ = torch.matmul(self.OD.T, contagious_infects)
        contagious_toJ = torch.clamp(contagious_toJ, min=0.0)

        all_toJ = torch.matmul(self.OD.T, self.POP)
        contagious_ratio_toJ = contagious_toJ / all_toJ
        contagious_ratio_toJ = torch.clamp(contagious_ratio_toJ, min=0.0)

        lam = torch.matmul(self.OD, contagious_ratio_toJ * self.betas * (1 - 0.25 * action))
        lam = torch.clamp(lam, min=0.0)

        dS = -self.simState[:, 0] * lam
        dE = self.simState[:, 0] * lam - self.sigma * self.simState[:, 1]
        dI = self.sigma * self.simState[:, 1] - self.gamma * self.simState[:, 2]
        dR = self.gamma * self.simState[:, 2]


        # 计算新增事件
        addE = self.simState[:, 0] * lam
        self.daily_new_E[self.day - 2] = addE.sum()
        # self.daily_new_E.append(addE.sum().item())
        addI = self.sigma * self.simState[:, 1]
        self.daily_new_I[self.day - 2] = addI.sum()
        # 更新状态
        dState = torch.stack([dS, dE, dI, dR], dim=1)
        self.simState += dState
        self.simRes[self.day - 1] = self.simState

        # 构造state/obs
        if self.day >= self.WINDOW_SIZE:
            Is_window = self.simRes[(self.day-self.WINDOW_SIZE): self.day, :, :]
        else:
            padding = self.WINDOW_SIZE - self.day
            current_data = self.simRes[:self.day, :, 2]
            Is_window = torch.nn.functional.pad(current_data, (0, 0, padding, 0), "constant", 0)
        obs = Is_window.flatten() / 1e4

        # 计算奖励
        total_infections = self.simState[:, 2].sum()
        health_cost = -torch.clamp((total_infections - self.capacity) / self.capacity, min=0)
        economy_cost = -action.sum() / self.ZONE_NUM

        t_order_cost = -torch.abs(action - self.actions[-2]).sum() / self.ZONE_NUM
        s_order_cost = -self.spatio_entropy_adj_matrix(action, self.adj_matrix) / self.ZONE_NUM
        f_order_cost = -self.spatio_entropy_adj_matrix(action, self.flow_top3_matrix) / self.ZONE_NUM

        if self.city=='sz':
            a_order_cost = -self.spatio_entropy_adm(action, self.adm_matrix) / self.ZONE_NUM
        else:
            a_order_cost = torch.tensor(0, device=self.device)

        if self.reward_mode == 4:
            order_cost = torch.tensor(0, device=self.device)  # 无秩序-4
        elif self.reward_mode == 2:
            order_cost = t_order_cost * 2  # t时间秩序-2
        elif self.reward_mode == 3:
            order_cost = s_order_cost * 2  # s空间秩序-3
        elif self.reward_mode == 5:
            order_cost = f_order_cost * 2  # f秩序-5
        elif self.reward_mode == -1:
            order_cost = a_order_cost * 2 # a秩序- -1
        elif self.reward_mode == 1:
            order_cost = t_order_cost + s_order_cost  # st秩序-1
        elif self.reward_mode == 6:
            order_cost = t_order_cost + f_order_cost  # ft秩序-6        
        elif self.reward_mode == -2:
            order_cost = t_order_cost + a_order_cost # at秩序- -2
        

        reward = health_cost * 60 + economy_cost + order_cost * 2

        # 记录奖励，使用Tensor或转换为列表后处理
        self.history_cost['reward'][self.day - 2] = reward
        self.history_cost['sdo'][self.day - 2] = -s_order_cost
        self.history_cost['fdo'][self.day - 2] = -f_order_cost
        self.history_cost['ado'][self.day - 2] = -a_order_cost

        info = {}
        done = False
        if self.day > self.period:
            # 计算指标
            total_infections = self.simRes[:, :, 2].sum(dim=1)  # 总感染数
            ep_r = self.history_cost['reward'].sum()
            ep_overload = (total_infections.max() - self.capacity) / self.capacity
            ep_intensity = self.actions.sum()
            ep_sdo = self.history_cost['sdo'].sum()
            ep_fdo = self.history_cost['fdo'].sum()
            ep_tdo = torch.sum(torch.abs(self.actions[1:] - self.actions[:-1]), dim=0).sum()
            ep_ado = self.history_cost['ado'].sum()

            # 构造结果字典
            info = {
                'ep_r': ep_r.item(),
                'ep_overload': ep_overload.item(),
                'ep_intensity': ep_intensity.item(),
                'ep_sdo': ep_sdo.item(),
                'ep_fdo': ep_fdo.item(),
                'ep_tdo': ep_tdo.item(),
                'ep_ado': ep_ado.item()
            }

            done = True


        return obs, reward, done, info

    def render(self, title=''):

        plt.figure(dpi=120, figsize=(4.2, 3.6))
        plt.grid(linestyle='-.', axis='both')

        plt.plot(np.sum(self.simRes[:, :, 2], axis=1), label='Infections', color='orange', linewidth=2)

        plt.axhline(self.capacity, ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, 1e7)
        plt.ylim(0, self.ylim)

        plt.legend(fontsize=8, loc='upper left')
        plt.twinx()
        plt.plot(np.mean(self.actions, axis=1), label='NPI', color='green', linewidth=2)

        plt.plot(self.actions, alpha=0.3, color='0.5')

        plt.ylim(-1, 3)
        plt.yticks(range(0, 3, 1))

        plt.title("Daily Current Infection (" + self.city + ")")

        plt.legend(fontsize=8, loc='upper right')

        plt.show()


if __name__ == '__main__':


    from config import args
    env = EpidemicModelTensor(args)
    actions = np.ones((120, env.ZONE_NUM))
    actions = torch.from_numpy(actions).float().to(env.device)

    random.seed(3074)
    rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    rand_idxs = torch.tensor(rand_list).to(env.device)

    import time
    start_time = time.time()
    for i in range(60):
        env.reset(rand_idxs)
        ep_s = 0
        ep_r = 0
        while True:

            s_, r, done, info = env.step(action=actions[ep_s] * (i % 3))
            ep_s += 1
            ep_r += r

            if done:
                ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_ado, ep_tdo = info.values()
                # env.render()

                peak_day = torch.argmax(env.simRes[:, :, 2].sum(dim=1)).item()
                # print(torch.max(env.simRes[:, :, 2].sum(dim=1)))
                # print(env.simRes[:, :, 2].sum(dim=1))
                # print(env.history_cost['reward'])
                print(
                    "level %d||  reward:%.4f\t  overload:%.4f\t intensity:%.2f\t tdo:%.4f\t sdo:%.4f\t fdo:%.2f\t ado:%.2f\t peak_day:%d" % (
                        i, ep_r, ep_overload, ep_intensity, ep_tdo, ep_sdo, ep_fdo, ep_ado, peak_day))

                # np.save(f'res/simRes_{env.city}_{env.R0}_level_{i}.npy', np.array(env.simRes))

                break

    print("--- 耗时： %s seconds ---" % (time.time() - start_time))