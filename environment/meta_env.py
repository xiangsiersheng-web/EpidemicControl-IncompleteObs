# -*- ecoding: utf-8 -*-
# @ModuleName: meta_env
# @Function:
#  
# @Time: 2023/4/14 13:52
import random

import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle
from dynamic.env_change_rules.beta_change import BetaChangeRule


class EpidemicModel(gym.Env):
    def __init__(self, reward_mode=4, city='sz', R0='high'):
        self.reward_mode = reward_mode
        self.city = city
        self.R0 = R0
        # 子区域邻接关系
        with open('data/' + self.city + '/adj_dict.pkl', 'rb') as file:
            self.adj_dict = pickle.load(file)

        with open('data/' + self.city + '/flow_top3_dict.pkl', 'rb') as file:
            self.flow_top3_dict = pickle.load(file)

        if self.city =='sz':
            with open('data/' + self.city + '/adm_dict.pkl', 'rb') as file:
                self.adm_dict = pickle.load(file)


        # 子区域间流动
        self.OD = np.load('data/' + self.city + '/flow.npy')

        # 子区域人口数量
        self.POP = np.load('data/' + self.city + '/population.npy')
        self.ZONE_NUM = self.POP.shape[0]

        # 传染病相关参数（感染者移动比例、beta、潜伏期、恢复期）
        self.Pm = 0.4
        self.beta = 0.8 if self.R0 == 'high' else 0.4

        self.betas = np.array([self.beta] * self.ZONE_NUM)
        self.sigma = 1 / 3
        self.gamma = 1 / 7

        # beta变化规则
        self.use_beta_change = False
        self.beta_matrix = None

        city_capacity = {
            'sz': 4e6,
            'tokyo': 2.2e6,
            'nyc': 2e6,
            'sh': 5.5e6
        }

        city_ylim = {
            'sz': 9e6,
            'tokyo': 6e6,
            'nyc': 6e6,
            'sh': 1.2e7
        }

        self.capacity = city_capacity[self.city]
        self.capacity = self.capacity if self.R0 == 'high' else self.capacity / 2

        self.ylim = city_ylim[self.city]
        # 模拟周期
        self.period = 120

        # 状态的观测天数
        self.WINDOW_SIZE = 7

        self.reset()


    def set_init_params(self, args):
        """需要修改环境参数时，调用此函数，会覆盖掉init中已经初始化的参数"""
        args_dict = vars(args)
        # 传染病相关参数（感染者移动比例、beta、潜伏期、恢复期）
        self.Pm = 0.4 if "ODE_Pm" not in args_dict else args.ODE_Pm
        self.beta = 0.8 if "ODE_beta" not in args_dict else args.ODE_beta
        if args.R0 == 'high' and args.ODE_beta != 0.8:
            print('!!! warn: beta should be 0.8 when R0 is high')

        self.betas = np.array([self.beta] * self.ZONE_NUM)
        self.sigma = 1 / 3 if "ODE_sigma" not in args_dict else args.ODE_sigma
        self.gamma = 1 / 7 if "ODE_gamma" not in args_dict else args.ODE_gamma

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

    def set_dynamic_env_params(self, args):
        print("动态环境为：", args.beta_change_rule)
        if args.beta_change_rule != BetaChangeRule.NONE:
            self.use_beta_change = True
            self.beta_matrix = args.beta_matrix

    def spatio_entropy(self, action, adj_dict):
        """
        计算空间熵
        :param action: 同一时间，各子区域动作
        :param adj_dict: 子区域邻接关系
        :return: 空间熵
        """
        action = action.reshape(self.ZONE_NUM)
        spatial_ent = 0
        for k, v in adj_dict.items():
            if type(k) == str:
                sample = action[v]
                w = len(action[v])
            else:
                sample = action[[k, *v]]
                w = 1
            values, counts = np.unique(sample, return_counts=True)
            probabilities = counts / np.sum(counts)
            entropy = -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 1 else 0

            spatial_ent += (entropy*w)

        return spatial_ent

    def reset(self, rand_list=None):
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
        self.actions = [np.zeros(self.ZONE_NUM)]
        # self.actions = [np.zeros((1, self.ZONE_NUM))]
        self.history_cost = {"reward": [], "sdo": [], "fdo": [], "ado": []}

        self.daily_new_E = []
        self.daily_new_I = []
        self.daily_attack_rate = []

        self.simState = np.zeros((self.ZONE_NUM, 4))
        self.simState[:, 0] = self.POP

        self.set_init_seed(rand_list=rand_list)

        self.simRes = [self.simState]

        # pad用于填充状态数组，在self.day<self.WINDOW_SIZE时，填充0
        Is_window = np.pad(np.array(self.simRes)[:, :, 2], ((self.WINDOW_SIZE - self.day, 0), (0, 0)), 'constant',
                           constant_values=(0, 0))

        obs = Is_window.flatten()
        return obs

    def set_init_seed(self, init_infection=100, rand_list=None):
        """
        设置初始感染者种子。

        此函数用于：
        - 设置随机种子 random.seed(3074)。
        - 随机选择初始感染者位置。
        - 将初始感染者数量添加到初始状态中。
        """
        if rand_list is None:
            random.seed(3074)
            rand_list = [random.randint(0, self.ZONE_NUM - 1) for _ in range(init_infection)]

        for sid in rand_list:
            self.simState[sid, 1] += 1
            self.simState[sid, 0] -= 1

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
        if self.use_beta_change:
            self.betas = self.beta_matrix[self.day - 1]
        self.day += 1
        self.actions.append(action)

        # 仓室模型step

        contagious_infects = self.simState[:, 1] + self.Pm * self.simState[:, 2]

        contagious_toJ = np.sum(self.OD.T * contagious_infects, axis=1)
        contagious_toJ[contagious_toJ < 0.0] = 0.0

        all_toJ = np.sum(self.POP * self.OD.T, axis=1)
        contagious_ratio_toJ = contagious_toJ / all_toJ
        contagious_ratio_toJ[contagious_ratio_toJ < 0.0] = 0.0

        lam = np.sum(self.OD * contagious_ratio_toJ * self.betas * (1 - 0.25 * action), axis=1)
        lam[lam < 0.0] = 0.0

        dS = -self.simState[:, 0] * lam
        dE = self.simState[:, 0] * lam - self.sigma * self.simState[:, 1]
        dI = self.sigma * self.simState[:, 1] - self.gamma * self.simState[:, 2]
        dR = self.gamma * self.simState[:, 2]

        dState = [dS, dE, dI, dR]
        addE = self.simState[:, 0] * lam
        self.daily_new_E.append([addE.sum()])
        addI = self.sigma * self.simState[:, 1]
        self.daily_new_I.append(addI.sum())
        self.simState = self.simState + np.array(dState).T
        self.simRes.append(self.simState)

        # 构造state/obs

        if self.day >= self.WINDOW_SIZE:
            Is_window = np.array(self.simRes)[-self.WINDOW_SIZE:, :, 2]
        else:
            Is_window = np.pad(np.array(self.simRes)[:, :, 2], ((self.WINDOW_SIZE - self.day, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))

        obs = Is_window.flatten() / 1e4

        # 计算奖励
        total_infections = self.simState[:, 2].sum()

        health_cost = -max((total_infections - self.capacity) / self.capacity, 0)

        economy_cost = -action.sum() / self.ZONE_NUM

        t_order_cost = -np.abs(action - self.actions[-2]).sum() / self.ZONE_NUM # 时间秩序, 当前行动与上一个行动之间的差异
        s_order_cost = -self.spatio_entropy(action, self.adj_dict) / self.ZONE_NUM
        f_order_cost = -self.spatio_entropy(action, self.flow_top3_dict) / self.ZONE_NUM

        if self.city=='sz':
            a_order_cost = -self.spatio_entropy(action, self.adm_dict) / self.ZONE_NUM
        else:
            a_order_cost=0

        if self.reward_mode == 4:
            order_cost = 0  # 无秩序-4
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

        self.history_cost['reward'].append(reward)
        self.history_cost['sdo'].append(-s_order_cost)
        self.history_cost['fdo'].append(-f_order_cost)
        self.history_cost['ado'].append(-a_order_cost)

        info = {}
        done = False
        if self.day > self.period:
            self.simRes = np.array(self.simRes)
            self.actions = np.array(self.actions)

            episode_metrics = {
                'ep_r': np.sum(self.history_cost['reward']),
                'ep_overload': (np.max(np.sum(self.simRes[:, :, 2], axis=1)) - self.capacity) / self.capacity,
                'ep_intensity': np.sum(self.actions),
                'ep_sdo': np.sum(self.history_cost['sdo']),
                'ep_fdo': np.sum(self.history_cost['fdo']),
                'ep_tdo': np.sum(abs(self.actions[1:] - self.actions[:-1]), axis=0).sum(),
                'ep_ado': np.sum(self.history_cost['ado'])
            }
            info = episode_metrics

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

        return



if __name__ == '__main__':
    # # 1.测试耗时
    # cities = ['sz', 'tokyo', 'nyc', 'sh']
    # env = EpidemicModel(city=cities[0], reward_mode=4, R0='high')
    # actions = np.ones((120, env.ZONE_NUM))
    #
    # random.seed(3074)
    # rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    #
    # import time
    # start_time = time.time()
    # for i in range(60):
    #     env.reset(rand_list=rand_list)
    #     ep_s = 0
    #     ep_r = 0
    #     while True:
    #
    #         s_, r, done, info = env.step(action=actions[ep_s] * (i % 3))
    #         ep_s += 1
    #         ep_r += r
    #
    #         if done:
    #             ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_ado, ep_tdo = info.values()
    #             # env.render()
    #             #
    #             peak_day = np.argmax(np.sum(env.simRes[:, :, 2], axis=1))
    #             # print(np.max(np.sum(env.simRes[:, :, 2], axis=1)))
    #             # print(np.sum(env.simRes[:, :, 2], axis=1))
    #             # print(env.history_cost['reward'])
    #             print(
    #                 "level %d||  reward:%d\t  overload:%.4f\t intensity:%.2f\t tdo:%.4f\t sdo:%.4f\t fdo:%.2f\t ado:%.2f\t peak_day:%d" % (
    #                     i, ep_r, ep_overload, ep_intensity, ep_tdo, ep_sdo, ep_fdo, ep_ado, peak_day))
    #
    #             # np.save(f'res/simRes_{env.city}_{env.R0}_level_{i}.npy', np.array(env.simRes))
    #
    #             break
    # print("--- 耗时： %s seconds ---" % (time.time() - start_time))

    # 2.测试一致性
    env = EpidemicModel(city='sz', reward_mode=4, R0='high')
    actions = np.ones((120, env.ZONE_NUM)) * 2
    np.random.seed(3047)
    action1 = np.random.randint(0, 3, env.ZONE_NUM)
    np.random.seed(305)
    action2 = np.random.randint(0, 3, env.ZONE_NUM)
    actions[5:10, 0:10] = action1[0:10]
    actions[10:15, 10:20] = action2[10:20]

    random.seed(3074)
    rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]

    import time

    start_time = time.time()
    for i in range(30):
        env.reset(rand_list=rand_list)
        ep_s = 0
        ep_r = 0
        while True:

            s_, r, done, info = env.step(action=actions[ep_s])
            ep_s += 1
            ep_r += r

            if done:
                ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_ado, ep_tdo = info.values()
                env.render()
                #
                peak_day = np.argmax(np.sum(env.simRes[:, :, 2], axis=1))
                total_new_I = sum(env.daily_new_I)
                daily_curr_I = np.sum(env.simRes[:, :, 2], axis=1)
                print(f"总新增感染人数：{total_new_I}，现存感染最大峰值：{max(daily_curr_I)}")
                # print(np.max(np.sum(env.simRes[:, :, 2], axis=1)))
                # print(np.sum(env.simRes[:, :, 2], axis=1))
                # print(env.history_cost['reward'])
                print(
                    "level %d||  reward:%.4f\t  overload:%.4f\t intensity:%.2f\t tdo:%.4f\t sdo:%.4f\t fdo:%.2f\t ado:%.2f\t peak_day:%d" % (
                        i, ep_r, ep_overload, ep_intensity, ep_tdo, ep_sdo, ep_fdo, ep_ado, peak_day))

                # np.save(f'res/simRes_{env.city}_{env.R0}_level_{i}.npy', np.array(env.simRes))

                break
    print("--- 耗时： %s seconds ---" % (time.time() - start_time))