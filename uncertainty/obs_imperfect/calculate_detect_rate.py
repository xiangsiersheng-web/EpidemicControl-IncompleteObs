import torch

from environment.uncertain_seir_vector import EpidemicModel, action_to_u
from uncertainty.obs_imperfect.main_rebuild import plot_one_env


def estimate_current_I(prev_I_obs, E, R, prev_OD, a, env):
    # Extract parameters from the environment
    beta = env.betas[0]  # (ZONE_NUM)
    sigma = env.sigma  # scalar
    gamma = env.gamma  # scalar
    Pm = env.Pm  # scalar
    POP = env.POP[0]  # (ZONE_NUM)
    S = env.POP[0] - E - prev_I_obs - R  # (ZONE_NUM)
    u_beta, u_in, u_out = env._action_to_u(a)  # (env_count, ZONE_NUM)

    # Compute contagious individuals (E + Pm * I)
    contagious_infects = E + Pm * prev_I_obs  # (ZONE_NUM, 1)

    # Total population moving to each zone
    all_toJ = prev_OD.T @ POP  # (ZONE_NUM, 1)

    # Total contagious population moving to each zone
    contagious_toJ = prev_OD.T @ contagious_infects  # (ZONE_NUM, 1)

    # Avoid division by zero
    all_toJ = torch.clamp(all_toJ, min=1e-6)

    # Compute the ratio of contagious individuals
    contagious_ratio_toJ = contagious_toJ / all_toJ
    contagious_ratio_toJ = torch.clamp(contagious_ratio_toJ, min=0.0)

    # Modified infection ratio with beta
    modified_ratio = contagious_ratio_toJ * beta * (1 - u_beta[0])  # (ZONE_NUM, 1)

    # Infection rate λ
    lam = prev_OD @ modified_ratio  # (ZONE_NUM, 1)
    lam = torch.clamp(lam, min=0.0)

    # SEIR model differential equations
    dS = -S * lam
    dE = S * lam - sigma * E
    dI = sigma * E - gamma * prev_I_obs
    dR = gamma * prev_I_obs

    # Update states
    S_new = S + dS
    E_new = E + dE
    I_new = prev_I_obs + dI
    R_new = R + dR

    return I_new, lam


def main(args):
    # 初始化环境
    args.action_dim = action_to_u.shape[0]
    args.env_data_dir = '../../data/'
    args.I_obs_imperfect = True
    args.I_obs_imperfect_down = 0.1
    args.I_obs_imperfect_up = 0.3
    env = EpidemicModel(args, env_count=1)
    s = env.reset()

    estimate_res = torch.zeros_like(env.simRes, device=env.device)
    estimate_rate = torch.ones(env.ZONE_NUM).to(env.device)

    done = False
    while not done:
        a = torch.randint(0, env.action_max, (env.env_count, env.ZONE_NUM), device=env.device)
        s_, r, done, _ = env.step(a)
        s = s_

        curr_I_true = env.simRes[0, env.day-1, :, 2]
        E = estimate_res[0, env.day-2, :, 1]
        R = estimate_res[0, env.day-2, :, 3]
        # E = env.simRes[0, env.day - 2, :, 1]
        # R = estimate_res[0, env.day - 2, :, 3]
        curr_I_obs = env.imperfect_states[0, env.day - 1, :, 2]
        detect_rate = curr_I_obs / curr_I_true
        # print("detect rate:", detect_rate)
        prev_I_obs = env.imperfect_states[0, env.day - 2, :, 2]
        prev_OD = env.new_OD[0]

        # estimate_rate += 0.1
        # estimate_rate = torch.clamp(estimate_rate, min=0.1, max=1.0)
        estimate_rate = 0.9 * estimate_rate + 0.1 * torch.ones(env.ZONE_NUM).to(env.device)
        while True:
            prev_I_obs_estimate = prev_I_obs / estimate_rate
            estimate_curr_I, lam = estimate_current_I(prev_I_obs_estimate, E, R, prev_OD, a, env)
            estimate_rate_new = curr_I_obs / (estimate_curr_I + 1e-8)
            estimate_rate_new = torch.clamp(estimate_rate_new, min=0.1, max=1.0)
            if (estimate_rate_new.mean() > estimate_rate.mean() - 1e-4):
                break
            estimate_rate = estimate_rate_new
        print(f'{env.day-1}:\n', (estimate_rate - detect_rate).mean())
        # print(estimate_rate - detect_rate)
        estimate_res[0, env.day-1, :, 2] = curr_I_obs / estimate_rate
        # 对estimate进行一次状态转移
        dI = estimate_res[0, env.day-1, :, 2] - estimate_res[0, env.day-2, :, 2]
        dR = env.gamma * estimate_res[0, env.day - 2, :, 2]
        estimate_res[0, env.day - 1, :, 3] = estimate_res[0, env.day - 2, :, 3] + dR
        if env.day < 20:
            estimate_res[0, env.day - 1, :, 1] = estimate_res[0, env.day - 1, :, 2]
        else:
            estimate_res[0, env.day - 1, :, 1] = estimate_res[0, env.day - 2, :, 1] + lam * estimate_res[0, env.day - 2, :, 0] - env.sigma * estimate_res[0, env.day - 2, :, 1]
        estimate_res[0, env.day - 1, :, 1] = torch.clamp(estimate_res[0, env.day - 1, :, 1], min=0.0)
        estimate_res[0, env.day - 1, :, 0] = env.POP[0] - estimate_res[0, env.day - 1, :, 1:].sum()

    time_steps = range(env.period + 1)
    plot_one_env(time_steps, env.imperfect_states[:, :, :, 2], env.simRes[:, :, :, 2], estimate_res[:, :, :, 2], env_index=0)
        # 前一天的seir、OD、传播参数，今天观测的seir，今天实际的seir
    print(env.imperfect_states.shape, env.simRes.shape, env.I_detect_rate.shape)
    print(env.imperfect_states[0, :, :, 2].sum(dim=0) / env.simRes[0, :, :, 2].sum(dim=0))
    print(env.I_detect_rate[0])



if __name__ == '__main__':
    from config import args
    args.device_name = 'cpu'
    main(args)