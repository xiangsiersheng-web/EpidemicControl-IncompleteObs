from environment.uncertain_seir_vector_v3 import EpidemicModel, action_to_u0, action_to_u1
import torch

class ExpertPolicy:
    def __init__(self, args, env):
        self.action_dim = args.action_dim
        self.WINDOW_SIZE = args.WINDOW_SIZE
        self.zone_num = args.zone_num
        self.state_contain_action = args.state_contain_action

        # 状态缩放尺度
        self.state_standard_scale = args.state_standard_scale

        self.POP = env.POP[0] # (zone_num,)
        self.capacity = env.capacity
        self.capacity_rate = self.capacity / self.POP.sum() # (zone_num,)

        self.save_s_a = False
        self.states = []
        self.actions = []

    def choose_action(self, state):
        """根据状态选择动作
        :param state: (batch_size, WINDOW_SIZE, zone_num, local_obs_dim * 2)
        :return action: (batch_size, zone_num)
        """
        # action初始为动作0
        # 1.如果前2天的感染者数量均大于10，则选择动作1
        # 2.如果前一天的感染者数量大于该区域人数的0.1capacity_rate%，则选择动作2
        # 3.如果前一天的感染者数量大于该区域人数的0.3capacity_rate%，则选择动作3
        # 4.如果前一天的感染者数量大于该区域人数的0.6capacity_rate%，则选择动作4
        I_s = state[:, :, :, :4].sum(dim=-1) # (B, WINDOW_SIZE, zone_num)
        I_num = I_s * self.POP.unsqueeze(0).unsqueeze(0)
        # I_num = I_s
        batch_size = state.shape[0]
        device = state.device  # 获取设备信息（CPU或GPU）

        # 获取前一天和前两天的感染者数量
        last_day_infected = I_num[:, -1, :]  # 前一天感染者数量，形状：(batch_size, zone_num)
        prev_day_infected = I_num[:, -2, :]  # 前两天感染者数量，形状：(batch_size, zone_num)
        mean_day_infected = (last_day_infected + prev_day_infected) / 2

        # 初始化动作为0
        action = torch.ones((batch_size, self.zone_num), dtype=torch.long, device=device)

        # 计算各个规则的阈值
        threshold_rule2 = 0.0001 * self.POP  # (zone_num,)
        threshold_rule3 = 0.0005 * self.POP
        threshold_rule4 = 0.001 * self.POP

        # 扩展阈值以匹配批量维度
        threshold_rule2 = threshold_rule2.unsqueeze(0)  # (1, zone_num)
        threshold_rule3 = threshold_rule3.unsqueeze(0)
        threshold_rule4 = threshold_rule4.unsqueeze(0)

        # 规则1：如果前2天的感染者数量均大于10，则选择动作1
        # mask_rule1 = (last_day_infected > 0.5) & (prev_day_infected > 0.5)
        mask_rule1 = (last_day_infected > 0)
        # 输出大于0的索引列表
        idx_rule1 = torch.nonzero(mask_rule1[0], as_tuple=False).squeeze(-1)
        # print(idx_rule1)
        action = torch.where(mask_rule1, torch.tensor(4, device=device), action)

        # 规则2：如果前一天的感染者数量大于阈值2，则选择动作2
        # mask_rule2 = last_day_infected > threshold_rule2
        # action = torch.where(mask_rule2, torch.tensor(2, device=device), action)
        #
        # # 规则3：如果前一天的感染者数量大于阈值2，则选择动作3
        # mask_rule3 = last_day_infected > threshold_rule3
        # action = torch.where(mask_rule3, torch.tensor(3, device=device), action)
        #
        # # 规则4：如果前一天的感染者数量大于阈值4，则选择动作4
        # mask_rule4 = last_day_infected > threshold_rule4
        # action = torch.where(mask_rule4, torch.tensor(4, device=device), action)

        # 记录状态和动作（可选）
        if self.save_s_a:
            self.states.append(state)
            self.actions.append(action)

        action = action.unsqueeze(-1).repeat(1, 1, 2)

        return action


def main():
    from config import args
    args.simulate_scale = 'community'
    args.zone_num = 643

    args.local_reward_weight = 0
    args.reward_weights = [1, 1 / 8 / 7, 1]

    args.local_obs_dim = 2
    args.state_contain_detected = True
    args.state_contain_detect_rate = True
    args.local_obs_dim = 5

    args.ODE_detect_E_rate = 1.0
    args.ODE_detect_I_rate = 1.0

    args.device_name = 'cpu'
    args.env_data_dir = '../../data/'
    args.action_dim = action_to_u0.shape[0] * action_to_u1.shape[0]

    args.state_is_sequence = True
    env = EpidemicModel(args, env_count=10)
    env.seed(3047)
    expert_policy = ExpertPolicy(args, env)
    expert_policy.save_s_a = True

    # for beta in np.arange(1.1, 0.3, -0.05):
    for beta in [0.8]:
        # args.use_beta_change = True
        args.ODE_beta = beta
        # args.use_reward_shaping = True
        # args.ODE_period = 50
        env = EpidemicModel(args, env_count=2)
        # env.dynamic_beta_change(type='spatial', std=0.1)
        s = env.reset()

        while True:
            a = expert_policy.choose_action(s)
            # if env.day in range(5,25):
            #     a = torch.ones(a.shape, dtype=torch.long, device=a.device) * 5
            # a = torch.ones(a.shape, dtype=torch.long, device=a.device) * 3

            s_, r, done, info = env.step(a)
            s = s_

            if done:
                env.render()
                env.render_all_rooms()

                e_r = info['ep_r'].mean(dim=0)
                total_infections = info['total_infections'].mean(dim=0)
                total_test_num = info['total_test_num'].mean(dim=0)
                total_quarantine = info['total_quarantine'].mean(dim=0)
                score = info['score'].mean(dim=0)
                print("e_r:", e_r, "total_infections:", total_infections, "total_test_num:", total_test_num,
                      "total_quarantine:", total_quarantine, "score:", score)
                # env.render()
                env.render()
                env.render_all_rooms()
                # 总感染的峰值天数需要对每个批次进行计算
                daily_new_I = env.daily_new_I.sum(dim=-1).cpu().numpy()
                daily_curr_I = env.simRes[:, :, :, 2].sum(dim=2)
                peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

                for batch_index in range(env.env_count):
                    total_new_I = sum(daily_new_I[batch_index])
                    print(
                        f"总新增感染人数：{total_new_I}，现存感染最大峰值：{daily_curr_I[batch_index, peak_days[batch_index]]:.8f}")
                    print(
                        f"Batch {batch_index}, level {0} || reward: {ep_r[batch_index]:.4f}\t"
                        f"overload: {ep_overload[batch_index]:.4f}\t intensity: {ep_intensity[batch_index]:.2f}\t"
                        f"tdo: {ep_tdo[batch_index]:.4f}\t sdo: {ep_sdo[batch_index]:.4f}\t"
                        f"fdo: {ep_fdo[batch_index]:.2f}\t ado: {ep_ado[batch_index]:.2f}\t"
                        f"peak_day: {peak_days[batch_index]}"
                    )

                break

    # 将专家策略产生的数据转换为张量
    expert_states = torch.cat(expert_policy.states, dim=0)  # 形状：(total_samples, WINDOW_SIZE, zone_num)
    expert_actions = torch.cat(expert_policy.actions, dim=0)  # 形状：(total_samples, zone_num)

    # 保存数据到文件
    torch.save({'states': expert_states, 'actions': expert_actions}, 'expert_data.pth')
    print("Data saved successfully! total samples: ", expert_states.shape[0])


if __name__ == '__main__':
    main()