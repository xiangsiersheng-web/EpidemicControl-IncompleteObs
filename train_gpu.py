import torch
import numpy as np
import pandas as pd

from algorithm.behavioral_clone.expert_policy import ExpertPolicy
from utils.normalization import Normalization, RewardScaling
from utils.replaybuffer_tensor import ReplayBufferTensor
from algorithm.ppo_discrete_gpu import PPO_discrete_gpu
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import logging
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
from uncertainty.obs_imperfect.generic_predictor import GenericPredictor
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

def build_tensorboard_path(args):
    writepath = 'runs/gpu/'
    if args.rl_type == 'uncertainty':
        writepath += 'uncertainty/'
        if args.use_obs_imperfect:
            writepath += 'obs_imperfect/'
        if args.gather_to_some_region:
            writepath += 'unsteady/'
        if args.use_beta_change or args.use_action_uncertainty:
            writepath += 'state_transition_uncertain/'

    if args.actor_critic_model == "lstm":
        writepath += 'lstm/'
    elif args.actor_critic_model == "gru":
        writepath += 'gru/'

    # 城市等信息
    writepath += args.city + args.timenow + str(args.experiment_idx) + "_" + args.R0

    if args.use_asymmetric:
        writepath += "_asymmetric"

    if os.path.exists(writepath): shutil.rmtree(writepath)

    return writepath



def evaluate_policy(args, env, agent, state_norm=None, evaluate_num=0,
                    generic_predictor = None, train_dl_in_eval = False):
    """
    评估模型：从初始状态开始，走完num_episodes条完整的轨迹
    """

    # 如果使用beta_change, 就需要重置beta_matrix
    if args.use_beta_change:
        env.dynamic_beta_change(type=args.test_beta_change_rule, std=0.2)
    s = env.reset()
    if evaluate_num > START_TRAIN_DL or evaluate_num == -1:
        s = generic_predictor.predict(env=env, s=s)  # 根据env的状态进行一个预测
    done = False

    while not done:
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
        # a = args.expert_policy.choose_action(s)
        s_, r, done, info = env.step(a)
        if evaluate_num > START_TRAIN_DL or evaluate_num == -1:
            s_ = generic_predictor.predict(env=env, s=s_)  # 根据env的状态进行一个预测

        s = s_

    fig_dir = None if 'eval_fig_dir' not in vars(args) else args.eval_fig_dir
    if args.show_fig:
        # env.render_all_rooms()
        env.render(title="eval:" + str(evaluate_num))
        generic_predictor.render(title="eval:" + str(evaluate_num) + "(rebuild)")
        # for i in [0, 10, 30, 50, 80, 100, 200]:
        # for i in range(20):
        #     env.render_one_region(region_idx=i)
    elif fig_dir is not None:
        env.render(title="eval:" + str(evaluate_num), fig_dir=fig_dir)

    # 在评估环境中训练
    if train_dl_in_eval:
        generic_predictor.train(env)

    e_r = info['ep_r'].mean(dim=0).item()
    total_infections = info['total_infections'].mean(dim=0).item()
    total_test_num = info['total_test_num'].mean(dim=0).item()
    total_quara_num = info['total_quara_num'].mean(dim=0).item()
    score = info['score'].mean(dim=0).item()

    return e_r, total_infections, total_test_num, total_quara_num, score, info


def my_test(args, seed=3047, env_count = 10):
    model_idx = args.model_idx
    args.env_count = env_count
    eval_env = EpidemicModel(args, env_count, is_evaluation=True)
    eval_env.seed(seed)
    args.expert_policy = ExpertPolicy(args, eval_env)
    agent = PPO_discrete_gpu(args)
    agent.load(model_idx)
    args.eval_fig_dir = agent.directory + '/' + str(model_idx)

    # 测试时也应该根据参数，初始化state_norm
    state_norm = None
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 初始化generic-predictor
    generic_predictor = GenericPredictor(args, agent.directory)
    generic_predictor.load(model_idx)

    e_r, total_infections, total_test_num, total_quara_num, score, info = evaluate_policy(args, eval_env, agent, state_norm, evaluate_num = -1, generic_predictor=generic_predictor)
    # print(
    #     f"Experiment {args.experiment_idx} \t Model: {args.model_idx} \t"
    #     f"Reward: {e_r:.2f} \t Overload: {e_overload * 100:.2f}% \t"
    #     f"Intensity: {e_intensity / eval_env.period / eval_env.ZONE_NUM:.3f} \t"
    #     f"Infection Rate: {e_infections_rate / eval_env.period * 100:.2f}% \t"
    #     f"Affected Rate: {e_affected_rate / eval_env.period * 100:.2f}% \t"
    #     f"TDO: {e_tdo / eval_env.period:.3f}"
    # )

    # np.save(f"res/model_{args.city}_{str(args.experiment_idx)}_{str(args.model_idx)}_{args.R0}_simRes.npy",
    #         np.array(eval_env.simRes))
    # np.save(f"res/model_{args.city}_{str(args.experiment_idx)}_{str(args.model_idx)}_{args.R0}_actions.npy",
    #         np.array(eval_env.actions))
    args.eval_fig_dir = None

    return eval_env, info


def main(args, seed = 3047):
    """
    用于训练的主函数，接收参数args，seed用于控制随机性
    """
    # 初始化环境
    args.env_count = int(args.batch_size / args.ODE_period)
    assert args.batch_size == args.env_count * args.ODE_period, "batch_size != env_count * ODE_period, 维度不匹配！"
    env = EpidemicModel(args, env_count=args.env_count)
    env_evaluate = EpidemicModel(args, env_count=args.env_count, is_evaluation=True)

    # 初始化专家策略
    args.expert_policy = ExpertPolicy(args, env)

    # 设置随机种子，使得训练可以复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    env.seed(seed)
    env_evaluate.seed(seed)

    # 时间戳
    timenow = str(datetime.now())[0:-7]
    timenow = '_' + timenow[0:10] + '_' + timenow[11:13] + '-' + timenow[14:16] + '-' + timenow[17:19] + '_'

    args.timenow = timenow
    args.max_episode_steps = env.period  # Maximum number of steps per episode

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    # 缓存池
    replay_buffer = ReplayBufferTensor(args)
    # 初始化agent
    agent = PPO_discrete_gpu(args)
    # 是否使用预训练模型
    if args.load_model:
        agent.load(args.model_idx)
        evaluate_num += args.model_idx

    # Build a tensorboard
    writepath = build_tensorboard_path(args)
    writer = SummaryWriter(log_dir=writepath)
    # 按step收集数据的字典，key为step，value为该step对应的所有指标
    step_data = {}

    # LOGGING
    log_path = os.path.join("logs", args.city + timenow + str(args.experiment_idx) + "_" + args.R0 + ".log")
    if not os.path.exists(f"logs"):
        os.makedirs(f"logs")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__), saving=True)
    logger.info(args) # 记录训练参数
    logger.info(
        f'task starts: {str(datetime.now())[0:-7]}') # 记录训练开始时间

    # 初始化预测模型
    generic_predictor = GenericPredictor(args, model_dir = agent.directory)

    # 记录模型文件和tensorboard位置
    logger.info(f"RL Model will be saved at {agent.directory}")
    logger.info(f"DL Model will be saved at {generic_predictor.predictor.model_path}")
    logger.info(f"Tensorboard will be saved at {writepath}")

    # 状态归一化器
    state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)  # Trick 2:state normalization. error: 这里好像有问题
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    if args.eval_before_train_start:
        # 评估模型
        e_r, total_infections, total_test_num, total_quara_num, score, _ = evaluate_policy(args, env_evaluate, agent, state_norm, evaluate_num=-1, generic_predictor = generic_predictor)
        # 按step存储评估数据
        if total_steps not in step_data:
            step_data[total_steps] = {}
        step_data[total_steps]['eval_num'] = evaluate_num
        step_data[total_steps]['reward'] = e_r
        step_data[total_steps]['total_infections'] = total_infections
        step_data[total_steps]['total_test_num'] = total_test_num
        step_data[total_steps]['total_quara_num'] = total_quara_num
        step_data[total_steps]['score'] = score

    while total_steps < args.max_train_steps:
        # 如果使用beta_change, 就需要重置beta_matrix
        if args.use_beta_change:
            env.dynamic_beta_change(type=args.train_beta_change_rule, std=0.1)

        s = env.reset() # todo：预测模型插入
        if evaluate_num > START_TRAIN_DL:
            s = generic_predictor.predict(env = env, s = s) # 根据env的状态进行一个预测
        # env.local_reward_weight = (1 - total_steps / args.max_train_steps) ** 2
        if args.use_state_norm:
            s = state_norm(s, update=True)
        if args.use_reward_scaling:
            reward_scaling.reset()

        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)        # todo：预测模型插入
            if evaluate_num > START_TRAIN_DL:
                s_ = generic_predictor.predict(env = env, s = s_)  # 根据env的状态进行一个预测

            if args.use_state_norm:
                s_ = state_norm(s_, update=True)

            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            if done and episode_steps != args.max_episode_steps:
                dw = True # dw 似乎始终为false，但计算gae时用的是done，所以应该还是没问题
            else:
                dw = False

            # 存入数据到缓存池
            for i in range(replay_buffer.env_count):
                replay_buffer.store(episode_steps - 1 + i * args.ODE_period, s[i], a[i], a_logprob[i], r[i], s_[i], dw, done)

            s = s_
            total_steps += args.env_count

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                # RL 训练，从缓存池获取数据更新actor-critic网络参数
                a_loss, c_loss, entropy, lr_now = agent.update(replay_buffer, total_steps)
                # DL 训练，从env中获取数据更新预测模型的参数
                if args.train_dl_in_train:
                    generic_predictor.train(env)

                replay_buffer.count = 0

                writer.add_scalar('train_loss/a_loss', a_loss.item(), global_step=total_steps)
                writer.add_scalar('train_loss/c_loss', c_loss.item(), global_step=total_steps)
                writer.add_scalar('train_loss/entropy', entropy.item(), global_step=total_steps)
                writer.add_scalar('train_loss/lr', lr_now, global_step=total_steps)

                # # 按step存储训练损失数据
                # step_data[total_steps] = {}
                # step_data[total_steps]['a_loss'] = a_loss.item()
                # step_data[total_steps]['c_loss'] = c_loss.item()
                # step_data[total_steps]['entropy'] = entropy.item()
                # step_data[total_steps]['lr'] = lr_now

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                train_dl_in_eval = evaluate_num >= START_TRAIN_DL
                if evaluate_num >= START_TRAIN_DL:
                    args.train_dl_in_train = False
                    logger.info(f"Adjust DL train way: train_dl_in_train={args.train_dl_in_train}, "
                                f"train_dl_in_eval={train_dl_in_eval}")

                e_r, total_infections, total_test_num, total_quara_num, score, _ = evaluate_policy(args, env_evaluate, agent,
                                                                                                 state_norm=state_norm, evaluate_num=evaluate_num,
                                                                                                 generic_predictor = generic_predictor,
                                                                                                 train_dl_in_eval = train_dl_in_eval)

                evaluate_rewards.append(e_r)

                writer.add_scalar('eval_Index/ep_r', e_r, global_step=total_steps)
                writer.add_scalar('eval_Index/total_infections', total_infections, global_step=total_steps)
                writer.add_scalar('eval_Index/total_test_num', total_test_num, global_step=total_steps)
                writer.add_scalar('eval_Index/total_quara_num', total_quara_num, global_step=total_steps)
                writer.add_scalar('eval_Index/score', score, global_step=total_steps)

                logger.info(
                    f"Eval {evaluate_num} \t Time: {datetime.now():%H:%M:%S} \t"
                    f"Reward: {e_r:.2f} \t info: {total_infections, total_test_num, total_quara_num, score}\t"
                )

                agent.save(evaluate_num)
                generic_predictor.save(evaluate_num + 1)    # DL 模型保存

                # 按step存储评估数据
                if total_steps not in step_data:
                    step_data[total_steps] = {}
                step_data[total_steps]['eval_num'] = evaluate_num
                step_data[total_steps]['reward'] = e_r
                step_data[total_steps]['total_infections'] = total_infections
                step_data[total_steps]['total_test_num'] = total_test_num
                step_data[total_steps]['total_quara_num'] = total_quara_num
                step_data[total_steps]['score'] = score

    sorted_steps = sorted(step_data.keys())
    data_list = []
    for step in sorted_steps:
        row = {'step': step}
        row.update(step_data[step])
        data_list.append(row)
    pd_data = pd.DataFrame(data_list)

    if args.use_state_norm:
        state_norm.save(agent.directory, 'state_norm.pth')
    for handler in logger.handlers:
        handler.close()
    logger.handlers = []
    logging.shutdown()

    return pd_data

def train_all_scenes():
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

def train_in_spatial_beta():
    args.batch_size = 6000
    args.mini_batch_size = 128
    args.device_name = 'cpu'
    args.experiment_idx = 4

    args.lr_a = 3e-4
    args.lr_c = 3e-4
    args.load_model = False
    args.max_train_steps = int(1.2e6)
    args.env_data_dir = "./data/"
    args.train_scenario_description = " 训练场景描述：空间变化场景下的训练，状态中包含beta，beta 符合均分分布：  noise = torch.rand(self.env_count, self.period, device=self.device) * 0.3 "

    args.use_beta_change = True
    args.train_beta_change_rule = "spatial"
    args.test_beta_change_rule = "spatial"
    args.state_contain_beta = True
    args.state_dim = 2 * args.state_dim  # 加上beta信息

    main(args)


def train_state_contain_action():
    pass


def train_in_imperfect():
    pass

def _config_args():
    args.batch_size = 1200
    args.mini_batch_size = 64

    args.max_train_steps = 2400 * 20
    args.evaluate_freq = 2400
    args.K_epochs = 4

    args.device_name = 'cuda:0'
    if not torch.cuda.is_available():
        args.device_name = 'cpu'

    args.lr_a = args.lr_a * args.mini_batch_size / 64
    args.lr_c = args.lr_a
    args.load_model = False

    args.env_data_dir = "./data/"
    args.action_dim = action_to_u0.shape[0] * action_to_u1.shape[0]
    # args.action_type = 1
    args.simulate_scale = 'community'
    args.zone_num = 654
    args.state_dim = args.zone_num
    args.hidden_width = 256
    args.WINDOW_SIZE = 3

    args.state_contain_detected = True
    args.state_contain_Q = True
    args.state_contain_R = True
    args.state_contain_detect_rate = True

    args.state_contain_detected = True
    args.state_contain_Q = False
    args.state_contain_R = False
    args.state_contain_detect_rate = False
    args.local_obs_dim = 2 + 2 * args.state_contain_detected + args.state_contain_Q + args.state_contain_R + args.state_contain_detect_rate

    args.actor_critic_model = 'mlp'
    # args.actor_critic_model = 'gru'
    args.local_reward_weight = 1
    args.daily_imported_cases = 0
    args.cost_ratio_test_to_quarantine = 1 / 8  # 检测成本与隔离成本的比率
    rw0 = 500
    rw2 = 1
    rw1 = rw2 * args.cost_ratio_test_to_quarantine / (1/args.ODE_sigma + 1/args.ODE_gamma)  # 1/80
    args.reward_weights = [rw0 * 1000, rw1 * 1000, rw2 * 1000]

    # args.reward_weights = [1000, 1/8/7, 1]

    # 减少lamda可以增加探索
    args.lamda = 0.9
    args.entropy_coef = 0.2

    # args.use_adv_norm = True
    args.use_state_norm = True

    # args.use_reward_shaping = True

    # 耦合预测模型
    args.predictor_type = 'none'
    args.train_dl_in_train = False
    args.train_dl_in_eval = False

    # 延迟隔离
    args.use_delay_quara = True

    # 奖励函数责任分发(默认启用奖励函数的责任分发)
    args.disable_response_distribution = False

    return args

START_TRAIN_DL: int = 5

def train_in_community():
    args.daily_imported_cases = 1
    args.max_train_steps = 2400 * 20
    args.show_fig = True

    main(args)
    args.model_idx = 18
    # args.show_fig = True
    env, _ = my_test(args, env_count=100)
    # # 保存到文件
    # import pickle
    # with open('draw/感染和动作的地图绘制/env.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开
    #     pickle.dump(env, f)

    # 在不确定环境下测试
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1
    my_test(args, env_count=100)

def train_in_uncertainty():
    args.daily_imported_cases = 1
    args.max_train_steps = 2400 * 20
    args.show_fig = True

    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1
    args.train_dl_in_train = False

    args.rl_type = 'uncertainty'
    for p_type in ['none', 'rebuild_gru_gnn_model_with_action', 'rebuild_gru_gnn_model_no_action']:
        args.predictor_type = p_type
        main(args)

        args.model_idx = 18
        my_test(args, env_count=100)

def compare_different_reward_func():
    """对比有无责任分发机制的奖励函数，将控制结果导出到csv文件中"""
    infos = []
    for drd in [True, False]:
        args.disable_response_distribution = drd
        args.use_obs_imperfect = False
        args.daily_imported_cases = 1
        args.max_train_steps = 2400 * 20
        args.show_fig = True

        main(args, seed=3047)
        args.model_idx = 18
        _, info = my_test(args, env_count=100)
        info['name'] = "certain_" + ("without_response_distribution" if drd else "with_response_distribution")
        infos.append(info)

        # 在不确定环境下测试
        args.use_obs_imperfect = True
        args.obs_imperfect_down = 0.1
        args.obs_imperfect_up = 1
        _, info = my_test(args, env_count=100)
        info['name'] = 'uncertain_' + ("without_response_distribution" if drd else "with_response_distribution")
        infos.append(info)
    # 导出表格
    processed_data = [
        {
            "name": info["name"],
            "ep_r": info["ep_r"].mean().item(),
            "total_infections": info["total_infections"].mean().item(),
            "total_test_num": info["total_test_num"].mean().item(),
            "total_quara_num": info["total_quara_num"].mean().item(),
            "total_control_cost": info["total_control_cost"].mean().item(),
            "score": info["score"].mean().item(),  # 计算均值并转为 Python 标量
            "score_avg": info["score_avg"].item(),
            "avg_zone_score": info["avg_zone_score"].mean().item(),
        }
        for info in infos
    ]

    # 生成 DataFrame
    df = pd.DataFrame(processed_data)

    csv_file_path = 'draw/消融实验-对比有无责任分发机制' + '/' + f'{args.R0}_control_result.csv'
    df.to_csv(csv_file_path, index=False)
    print("DataFrame has been exported to CSV: ", csv_file_path)

def compare_different_reward_func_plot():
    """
    对比有无责任分发机制的奖励函数表现，生成10个随机种子实验，收集奖励曲线并导出
    """
    # 1. 生成10个1000-9999的随机整数种子
    np.random.seed(42)  # 固定主种子确保实验可复现
    seeds = np.random.randint(1000, 9999, size=10)
    print(f"使用的随机种子: {seeds}")

    # 2. 初始化数据收集结构：存储所有实验的奖励曲线
    # 键格式: (是否启用责任分发机制, 种子), 值: 包含step和reward的DataFrame
    reward_data = {}

    # 3. 循环运行所有实验
    args.eval_before_train_start = True
    for seed in seeds:
        print(f"\n===== 开始种子 {seed} 的实验 =====")
        for drd in [True, False]:  # True: 无责任分发; False: 有责任分发
            # 设置当前实验参数
            args.disable_response_distribution = drd
            args.use_obs_imperfect = False
            args.daily_imported_cases = 1
            args.max_train_steps = 2400 * 20
            args.show_fig = False  # 训练过程不显示图，避免干扰

            # 运行主训练函数，获取奖励数据
            print(f"--- 责任分发机制: {'禁用' if drd else '启用'} ---")
            pd_data = main(args, seed=seed)  # main返回包含step和reward的DataFrame

            # 提取并存储奖励曲线（确保step和reward列存在）
            if 'step' in pd_data.columns and 'reward' in pd_data.columns:
                reward_curve = pd_data[['step', 'reward']].copy()
                reward_curve['seed'] = seed  # 标记种子
                reward_curve['drd'] = '无责任分发' if drd else '有责任分发'  # 标记机制类型
                reward_data[(drd, seed)] = reward_curve
            else:
                print(f"警告: 种子 {seed} (drd={drd}) 未返回有效奖励数据")

    # 4. 整理数据并保存
    all_rewards = pd.concat(reward_data.values(), ignore_index=True)
    save_dir = "draw/消融实验-对比有无责任分发机制"
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 生成带时间戳的文件名，避免重复
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"reward_data_{args.R0}_{timestamp}.csv")
    all_rewards.to_csv(save_path, index=False)
    print(f"实验数据已保存至: {save_path}")



if __name__ == '__main__':

    # training，师姐的原有训练函数
    # train_all_scenes()

    # 空间变化的beta下，训练智能体 - 2024.9.3
    # train_in_spatial_beta()

    # 状态包含action
    # train_state_contain_action()

    # 不完全观测影响
    # train_in_imperfect()

    _config_args()

    # 低R0实验
    # args.ODE_beta = 0.4
    # args.R0 = 'low'

    # 测试社区环境下的训练
    train_in_community()

    # 在不确定环境下训练
    # train_in_uncertainty()

    # # 对比奖励函数是否有责任分发机制的训练影响
    # for R0 in ['low', 'high']:
    #     args.R0 = R0
    #     args.ODE_beta = 0.8 if R0 == 'high' else 0.4
    #
    #     # compare_different_reward_func()
    #     compare_different_reward_func_plot()
