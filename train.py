import torch
import numpy as np

# from dynamic.env_change_rules.beta_change import BetaChangeRule
from utils.normalization import Normalization, RewardScaling
from utils.replaybuffer import ReplayBuffer
from utils.replaybuffer_asymmetric import ReplayBufferAsymmetric
from algorithm.ppo_discrete import PPO_discrete
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import logging
from environment.meta_env import EpidemicModel
from config import args

import warnings

warnings.filterwarnings("ignore")


def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    """
    初始化一个logger

    :param logpath: 日志文件保存路径
    :param filepath: 程序运行的绝对路径
    :param package_files: 
    :param displaying: 是否打印到控制台
    :param saving: 是否保存到文件
    :param debug: 是否打印debug信息
    """
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


def evaluate_policy(args, env, agent, state_norm):
    """
    评估模型：从初始状态开始，走完num_episodes条完整的轨迹
    """
    num_episodes = 1
    total_rewards, total_overload, total_intensity, total_sdo, total_fdo, total_tdo, total_ado = 0, 0, 0, 0, 0, 0, 0

    for _ in range(num_episodes):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        s_noise = s
        done = False

        history_obs = []
        while not done:

            a = agent.evaluate(s_noise)  # We use the deterministic policy during the evaluating
            s_, r, done, info = env.step(np.array(a))

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            if args.use_obs_noise:
                s_noise_, history_obs = add_noise_to_state(s_, history_obs, args, args.obs_noise_std)
            else:
                s_noise_ = s_

            s, s_noise = s_, s_noise_
        ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado = info.values()

        # env.render()

        total_rewards += ep_r
        total_overload += ep_overload
        total_intensity += ep_intensity
        total_sdo += ep_sdo
        total_fdo += ep_fdo
        total_tdo += ep_tdo
        total_ado += ep_ado

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


def my_test(args):
    model_idx = args.model_idx
    eval_env = EpidemicModel(reward_mode=args.experiment_idx, city=args.city, R0=args.R0)
    args.zone_num = eval_env.ZONE_NUM
    agent = PPO_discrete(args)
    agent.load(model_idx)

    for i in range(1):
        e_r, e_overload, e_intensity, e_sdo, e_fdo, e_tdo, e_ado = evaluate_policy(args, eval_env, agent,
                                                                                   args.use_state_norm)
        print(
            'experiment:%d\t model:%d\t reward:%.2f\t overload_ratio:%.2f%%\t intensity:%.3f\t sdo:%.3f tdo:%.3f fdo:%.3f ado:%.3f' %
            (args.experiment_idx, args.model_idx,
             e_r,
             e_overload * 100,
             e_intensity / eval_env.period / eval_env.ZONE_NUM,
             e_sdo / eval_env.period,
             e_tdo / eval_env.ZONE_NUM,
             e_fdo / eval_env.period,
             e_ado / eval_env.period)
        )

        np.save(f"res/model_{args.city}_{str(args.experiment_idx)}_{str(args.model_idx)}_{args.R0}_simRes.npy",
                np.array(eval_env.simRes))
        np.save(f"res/model_{args.city}_{str(args.experiment_idx)}_{str(args.model_idx)}_{args.R0}_actions.npy",
                np.array(eval_env.actions))
        eval_env.close()


def add_noise_to_state(s_, history_obs, args, relative_noise_std = 0.0):
    """对观测添加噪声，维护历史观测数据"""
    zone_num = int(len(s_) / args.WINDOW_SIZE)
    true_data = s_[-zone_num:]
    # 高斯噪声
    noise_std = relative_noise_std * (true_data.max() - true_data.min())
    noise = np.random.normal(0, noise_std, size=true_data.shape)
    noise_data = true_data + noise

    history_obs.append(noise_data)
    if len(history_obs) >= args.WINDOW_SIZE:
        obs = np.array(history_obs[-args.WINDOW_SIZE:])
    else:
        padding_days = args.WINDOW_SIZE - len(history_obs)
        padding = [np.zeros(zone_num) for _ in range(padding_days)]
        obs = np.array(padding + history_obs)
    s_noise_ = np.array(obs).flatten()
    return s_noise_, history_obs


def main(args, seed):
    """
    用于训练的主函数，接收参数args，seed用于控制随机性
    """
    env = EpidemicModel(reward_mode=args.experiment_idx, city=args.city, R0=args.R0)
    env_evaluate = EpidemicModel(reward_mode=args.experiment_idx, city=args.city, R0=args.R0)
    # Set random seed
    env.seed(seed)
    env_evaluate.seed(seed)
    env.set_dynamic_env_params(args) # 主要是设置 beta_matrix
    env_evaluate.set_dynamic_env_params(args)
    np.random.seed(seed)
    torch.manual_seed(seed)

    timenow = str(datetime.now())[0:-10]
    timenow = '_' + timenow[0:10] + '_' + timenow[-5:-3] + '-' + timenow[-2:] + '_'

    args.timenow = timenow
    args.max_episode_steps = env.period  # Maximum number of steps per episode
    args.zone_num = env.ZONE_NUM
    args.WINDOW_SIZE = env.WINDOW_SIZE

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    if args.use_asymmetric:
        replay_buffer = ReplayBufferAsymmetric(args)
    else:
        replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)
    # 使用预训练模型
    if args.load_model:
        agent.load(args.model_idx)
        evaluate_num += args.model_idx

    # Build a tensorboard
    writepath = 'runs/'
    writepath += args.city + timenow + str(args.experiment_idx) + "_" + args.R0
    if args.use_obs_noise:
        writepath += "_obsnoisestd=" + str(args.obs_noise_std)
    if args.use_asymmetric:
        writepath += "_asymmetric"
    if args.max_train_steps != 1.2e6:
        writepath += "_maxtrainsteps=" + str(args.max_train_steps)

    if os.path.exists(writepath): shutil.rmtree(writepath)
    writer = SummaryWriter(log_dir=writepath)

    # LOGGING
    log_path = os.path.join("logs", args.city + timenow + str(args.experiment_idx) + "_" + args.R0 + ".log")
    if not os.path.exists(f"logs"):
        os.makedirs(f"logs")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__), saving=True)
    logger.info(args) # 记录训练参数
    logger.info(
        f'task starts: {str(datetime.now())[0:-10]}') # 记录训练开始时间

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        s_noise = s
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        history_obs = []
        while not done:
            episode_steps += 1
            a, a_logprob, _ = agent.choose_action(s_noise)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(np.array(a))

            if args.use_state_norm:
                s_ = state_norm(s_)

            # 是否向s_添加噪声
            if args.use_obs_noise:
                s_noise_, history_obs = add_noise_to_state(s_, history_obs, args, args.obs_noise_std)
            else:
                s_noise_ = s_

            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            if args.use_asymmetric:
                # 如果使用非对称架构，就需要将真实s也存储下来
                replay_buffer.store(s, s_noise, a, a_logprob, r, s_, s_noise_, dw, done)
            else:
                replay_buffer.store(s_noise, a, a_logprob, r, s_noise_, dw, done)
            s, s_noise = s_, s_noise_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                if args.use_asymmetric:
                    a_loss, c_loss, entropy, lr_now = agent.update_asymmetric(replay_buffer, total_steps)
                else:
                    a_loss, c_loss, entropy, lr_now = agent.update(replay_buffer, total_steps)

                replay_buffer.count = 0

                writer.add_scalar('train_loss/a_loss', a_loss, global_step=total_steps)
                writer.add_scalar('train_loss/c_loss', c_loss, global_step=total_steps)
                writer.add_scalar('train_loss/entropy', entropy, global_step=total_steps)
                writer.add_scalar('train_loss/lr', lr_now, global_step=total_steps)

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1

                e_r, e_overload, e_intensity, e_sdo, e_fdo, e_tdo, e_ado = evaluate_policy(args,
                                                                                           env_evaluate,
                                                                                           agent,
                                                                                           state_norm)
                evaluate_rewards.append(e_r)

                writer.add_scalar('eval_Index/ep_r', e_r, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_overload', e_overload, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_intensity', e_intensity, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_sdo', e_sdo, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_tdo', e_tdo, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_fdo', e_fdo, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_ado', e_ado, global_step=total_steps)

                logger.info(
                    'evaluate_num:%d starts: %s\t reward:%.2f\t overload:%.2f%% \t intensity:%.3f\t sdo:%.3f\t tdo:%.3f\t fdo:%.3f\t ado:%.3f\t' %
                    (evaluate_num, str(datetime.now())[0:-10], e_r, e_overload * 100,
                     e_intensity / env.period / env.ZONE_NUM,
                     e_sdo / env.period,
                     e_tdo / env.ZONE_NUM,
                     e_fdo / env.period,
                     e_ado / env.period)
                ) # 记录日志：评估次数，评估时间，评估奖励，评估超载率，评估强度，评估sdo，评估tdo，评估fdo，评估ado

                agent.save(evaluate_num)

    for handler in logger.handlers:
        handler.close()
    logger.handlers = []
    logging.shutdown()

if __name__ == '__main__':
    # training，师姐的原有训练函数
    for city in ['sz','nyc','tokyo']:
        args.city = city
        for level in ['high']:
            args.R0 = level
            for mode in [4, 2, 3, 5, -1, 1, 6, -2]:
                # 4 corresponds to "basic"
                # 2 corresponds to "t-order"
                # 3, 5, -1 correspond to "s-order-adj", "s-order-mob", "s-order-adm", respectively
                # 1, 6, -2 correspond to "st-order-adj", "st-order-mob", "st-order-adm", respectively
                args.experiment_idx = mode

                args.lr_a = 3e-4
                args.lr_c = 3e-4
                args.load_model = False

                main(args, seed=3047)

    # # # 测试具有观测噪声时的训练过程 - 2024.8.22
    # args.city = "sz"
    # args.R0 = "high"
    # args.experiment_idx = 4  # base
    # for obs_noise_std in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    #     args.load_model = False
    #     args.lr_a = 3e-4
    #     args.lr_c = 3e-4
    #
    #     args.max_train_steps = int(1.2e4 * 100)
    #     if obs_noise_std == 0:
    #         args.use_obs_noise = False
    #         args.use_asymmetric = False
    #     else:
    #         args.use_obs_noise = True
    #         args.obs_noise_std = obs_noise_std
    #         args.use_asymmetric = True  # 使用非对称架构
    #     main(args, seed=3047)

    # # 测试动态环境下的智能体训练 - 2024.8.29
    # args.city = "sz"
    # args.R0 = "high"
    # args.experiment_idx = 4  # base
    # args.lr_a = 3e-4
    # args.lr_c = 3e-4
    # args.max_train_steps = int(1.2e4 * 100)
    # args.beta_change_rule = BetaChangeRule.TEMPORAL
    # # 构建beta_matrix
    # beta_matrix = np.full((args.ODE_period, args.zone_num), args.ODE_beta)
    # beta_temporal = [args.ODE_beta] * args.ODE_period
    # change_period = np.arange(10, 31, 1)
    # for i in change_period:
    #     beta_temporal[i] = 1.15 * args.ODE_beta
    #     beta_matrix[i] = beta_temporal[i]
    # args.beta_temporal = beta_temporal
    # args.beta_matrix = beta_matrix
    #
    # main(args, seed=3047)
