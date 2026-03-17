from environment.meta_env_tensor_vector import EpidemicModelTensorVector
import torch
import matplotlib.pyplot as plt
from config import args

args.device_name = "cpu"


def draw_I(env):
    plt.figure(dpi=120, figsize=(4.2, 3.6))
    plt.grid(linestyle='-.', axis='both')

    # 绘制所有批次的感染数量
    infections = env.simRes[:, :, :, 2].sum(dim=2)  # (env_count, period+1)

    # 绘制每个环境的感染曲线
    for i in range(infections.shape[0]):
        plt.plot(infections[i], label=f'Env {i + 1}', linewidth=1, alpha=0.5)  # 给每个环境一个标签

    plt.legend(fontsize=8, loc='upper right')  # 将图例位置改为左上角，以免挡住曲线
    plt.title('Infection curves for all environments')
    plt.xlabel('Time')
    plt.ylabel('Total Infections')
    plt.ylim(0, env.ylim)
    plt.show()


def spatial_beta():
    env = EpidemicModelTensorVector(args, env_count=100)
    zone_num = env.ZONE_NUM
    beta0 = env.beta
    env_count = env.env_count
    period = env.period
    stds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    actions = torch.zeros(env_count, period, zone_num)

    for std in stds:
        # 为每个zone生成beta_spatial
        beta_spatial = beta0 + std * torch.randn((zone_num,))
        # 生成env_count个环境的beta值
        env_betas = torch.zeros((env_count, zone_num))
        for i in range(env_count):
            env_betas[i] = beta_spatial[torch.randperm(zone_num)]

        # 扩展至env_count, env.period, zone_num的tensor
        beta_matrix = env_betas.unsqueeze(1).expand(-1, period, -1)

        env.adjust_beta_matrix(beta_matrix)
        env.reset()

        day = 0
        while True:
            s_, r, done, info = env.step(action=actions[:, day, :])
            day += 1
            if done:
                break
        # env.render(title=f'std={std}')
        # print(env.daily_new_I.sum(dim=1).mean(), env.daily_new_I.sum(dim=1).std())
        draw_I(env)

def temporal_beta():
    env = EpidemicModelTensorVector(args, env_count=50)
    zone_num = env.ZONE_NUM
    beta0 = env.beta
    env_count = env.env_count
    period = env.period
    stds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    actions = torch.zeros(env_count, period, zone_num)

    for std in stds:
        # 为每个zone生成beta_spatial
        beta_temporal = beta0 + std * torch.randn((period,))
        # 生成env_count个环境的beta值
        env_betas = torch.zeros((env_count, period))
        for i in range(env_count):
            env_betas[i] = beta_temporal[torch.randperm(period)]

        # 扩展至env_count, env.period, zone_num的tensor
        beta_matrix = env_betas.unsqueeze(2).expand(-1, -1, zone_num)

        env.adjust_beta_matrix(beta_matrix)
        env.reset()

        day = 0
        while True:
            s_, r, done, info = env.step(action=actions[:, day, :])
            day += 1
            if done:
                break
        env.render(title=f'std={std}')
        print(env.daily_new_I.sum(dim=1).mean(), env.daily_new_I.sum(dim=1).std())
        # draw_I(env)


def spatiotemporal_beta():
    env = EpidemicModelTensorVector(args, env_count=100)
    zone_num = env.ZONE_NUM
    beta0 = env.beta
    env_count = env.env_count
    period = env.period
    stds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    actions = torch.zeros(env_count, period, zone_num)

    for std in stds:

        beta_spatiotemporal = torch.full((period, zone_num), beta0)
        beta_spatiotemporal += std * torch.randn((period, zone_num))
        beta_matrix = torch.zeros(env_count, period, zone_num)
        for i in range(env_count):
            beta_matrix[i] = beta_spatiotemporal[torch.randperm(period), :]

        env.adjust_beta_matrix(beta_matrix)
        env.reset()

        day = 0
        while True:
            s_, r, done, info = env.step(action=actions[:, day, :])
            day += 1
            if done:
                break
        # env.render(title=f'std={std}')
        print(env.daily_new_I.sum(dim=1).mean(), env.daily_new_I.sum(dim=1).std())
        draw_I(env)


if __name__ == "__main__":
    spatial_beta()
    # temporal_beta()
    # spatiotemporal_beta()