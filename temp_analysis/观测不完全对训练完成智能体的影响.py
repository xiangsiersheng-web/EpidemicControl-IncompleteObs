from algorithm.ppo_discrete_gpu import PPO_discrete_gpu
from environment.meta_env_tensor_vector import EpidemicModelTensorVector
import warnings

warnings.filterwarnings("ignore")

from config import args


def evaluate_policy(args, env, agent, state_norm):
    """
    评估模型：从初始状态开始，走完num_episodes条完整的轨迹
    """
    num_episodes = 1
    total_rewards, total_overload, total_intensity, total_sdo, total_fdo, total_tdo, total_ado = 0, 0, 0, 0, 0, 0, 0

    for _ in range(num_episodes):
        # 如果使用beta_change, 就需要重置beta_matrix
        if args.train_beta_change_rule != 'none':
            env.dynamic_beta_change(type=args.test_beta_change_rule, std=0.2)
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False

        while not done:

            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating

            # if env.day in range(35, 42):
            #     env.adjust_contagious_OD(regions = [35], increase_ratio = [200])
            # if env.day in range(50, 55):
            #     a[:, 0] = 2

            s_, r, done, info = env.step(a)
            env.reset_contagious_OD()

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            s = s_
            # if env.day in [10, 20, 30, 40, 50, 60, 70, 90]:
            #     env.construct_I_detect_rate()
                # print('各个环境的均值 I_detect_rate:', env.I_detect_rate.mean(dim=0))
                # print("抽检第一个环境的检测率：", env.I_detect_rate[0, :10])
        ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado = info.values()


        total_rewards += ep_r.mean().item()
        total_overload += ep_overload.mean().item()
        total_intensity += ep_intensity.mean().item()
        total_sdo += ep_sdo.mean().item()
        total_fdo += ep_fdo.mean().item()
        total_tdo += ep_tdo.mean().item()
        total_ado += ep_ado.mean().item()

    average_reward = total_rewards / num_episodes
    average_overload = total_overload / num_episodes
    average_intensity = total_intensity / num_episodes
    average_sdo = total_sdo / num_episodes
    average_fdo = total_fdo / num_episodes
    average_tdo = total_tdo / num_episodes
    average_ado = total_ado / num_episodes

    return (
        average_reward,
        average_overload,
        average_intensity,
        average_sdo,
        average_fdo,
        average_tdo,
        average_ado
    )







args.device_name = "cpu"

# seed = 3047
# # 设置随机种子，使得训练可以复现
# np.random.seed(seed)
# torch.manual_seed(seed)

# 加载智能体
args.load_model = True
args.model_idx = 100
args.I_obs_imperfect = True
# args.actor_critic_model = "lstm"
# args.lstm_input_size = args.zone_num
# args.WINDOW_SIZE = 7
# args.lstm_hidden_size = 256
# args.hidden_width = 256

agent = PPO_discrete_gpu(args)
print(agent.directory)
agent.directory = '.' + agent.directory
if args.load_model:
    agent.load(args.model_idx)

# 定义环境
# args.ODE_beta = 0.8
args.I_obs_imperfect = True
args.env_data_dir = "../data/"
for up in [0.1, ]:
    args.I_obs_imperfect_down = 0.5
    args.I_obs_imperfect_up = 0.8
    env = EpidemicModelTensorVector(args, env_count=100)

    e_r, e_overload, e_intensity, e_sdo, e_fdo, e_tdo, e_ado = evaluate_policy(args,
                                                                               env,
                                                                               agent,
                                                                               None)
    print(e_r, e_overload, e_intensity, e_sdo, e_fdo, e_tdo, e_ado)
    env.render(title='Impact of imperfect observation'+str(up))
    # env.render_one_env(title='Impact of imperfect observation'+str(down))
    # env.render_one_region(title='Imperfect Observation'+str(up), region_idx=35)
    # for i in range(env.ZONE_NUM):
    #     env.render_one_region(env_idx=0, region_idx=i, title='Imperfect Observation (Region %d)' % i)
    # env.render_one_region(title='Impact of imperfect observation'+str(up), env_idx=2)


args.I_obs_imperfect = False
env0 = EpidemicModelTensorVector(args, env_count=100)
e_r, e_overload, e_intensity, e_sdo, e_fdo, e_tdo, e_ado = evaluate_policy(args,
                                                                            env0,
                                                                            agent,
                                                                            None)
print(e_r, e_overload, e_intensity, e_sdo, e_fdo, e_tdo, e_ado)
env0.render(title='perfect observation')
# env0.render_one_env(title='perfect observation')
# env0.render_one_region(title='Certain Scenario (Region 35)', region_idx=35)
# env0.render_one_region(title='Abnormal Transmission (Region 35)', region_idx=35)
# for i in range(env0.ZONE_NUM):
#     env0.render_one_region(env_idx=0, region_idx=i, title='Certain Scenario (Region %d)' % i)
# env0.render_one_region(env_idx=0, region_idx=0)
# env0.render_one_region(env_idx=0, region_idx=10)
# for i in range(env0.ZONE_NUM):
#     env.render_one_region(env_idx=i, title='Impact of imperfect observation' + str(down))
#     env0.render_one_region(env_idx=i, title='perfect observation')
