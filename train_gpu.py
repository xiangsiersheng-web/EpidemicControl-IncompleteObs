import torch
import numpy as np
import pandas as pd

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
    """Initialize a logger for training."""
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

    # City info
    writepath += args.city + args.timenow + str(args.experiment_idx) + "_" + args.R0

    if os.path.exists(writepath): shutil.rmtree(writepath)

    return writepath



def evaluate_policy(args, env, agent, state_norm=None, evaluate_num=0,
                    generic_predictor = None, train_dl_in_eval = False):
    """Evaluate model from initial state for a complete episode."""
    # Reset beta_matrix if using beta_change
    if args.use_beta_change:
        env.dynamic_beta_change(type=args.test_beta_change_rule, std=0.2)
    s = env.reset()
    if evaluate_num > START_TRAIN_DL or evaluate_num == -1:
        s = generic_predictor.predict(env=env, s=s)  # Predict state based on env
    done = False

    while not done:
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
        s_, r, done, info = env.step(a)
        if evaluate_num > START_TRAIN_DL or evaluate_num == -1:
            s_ = generic_predictor.predict(env=env, s=s_)  # Predict state

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

    # Train predictor during evaluation
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
    agent = PPO_discrete_gpu(args)
    agent.load(model_idx)
    args.eval_fig_dir = agent.directory + '/' + str(model_idx)

    # Initialize state_norm for testing if needed
    state_norm = None
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # Initialize generic-predictor
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
    """Main training function."""
    # Initialize environment
    args.env_count = int(args.batch_size / args.ODE_period)
    assert args.batch_size == args.env_count * args.ODE_period, "batch_size != env_count * ODE_period"
    env = EpidemicModel(args, env_count=args.env_count)
    env_evaluate = EpidemicModel(args, env_count=args.env_count, is_evaluation=True)

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    env.seed(seed)
    env_evaluate.seed(seed)

    # Timestamp
    timenow = str(datetime.now())[0:-7]
    timenow = '_' + timenow[0:10] + '_' + timenow[11:13] + '-' + timenow[14:16] + '-' + timenow[17:19] + '_'

    args.timenow = timenow
    args.max_episode_steps = env.period  # Maximum number of steps per episode

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    # Replay buffer
    replay_buffer = ReplayBufferTensor(args)
    # Initialize agent
    agent = PPO_discrete_gpu(args)
    # Load pretrained model
    if args.load_model:
        agent.load(args.model_idx)
        evaluate_num += args.model_idx

    # Build a tensorboard
    writepath = build_tensorboard_path(args)
    writer = SummaryWriter(log_dir=writepath)
    # Data collection dict

    step_data = {}

    # LOGGING
    log_path = os.path.join("logs", args.city + timenow + str(args.experiment_idx) + "_" + args.R0 + ".log")
    if not os.path.exists(f"logs"):
        os.makedirs(f"logs")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__), saving=True)
    logger.info(args)  # Log training params
    logger.info(
        f'task starts: {str(datetime.now())[0:-7]}')  # Log start time

    # Initialize predictor
    generic_predictor = GenericPredictor(args, model_dir = agent.directory)

    # Log model paths
    logger.info(f"RL Model will be saved at {agent.directory}")
    logger.info(f"DL Model will be saved at {generic_predictor.predictor.model_path}")
    logger.info(f"Tensorboard will be saved at {writepath}")

    # State normalizer
    state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    if args.eval_before_train_start:
        # Evaluate model
        e_r, total_infections, total_test_num, total_quara_num, score, _ = evaluate_policy(args, env_evaluate, agent, state_norm, evaluate_num=-1, generic_predictor = generic_predictor)
        # Store evaluation data by step
        if total_steps not in step_data:
            step_data[total_steps] = {}
        step_data[total_steps]['eval_num'] = evaluate_num
        step_data[total_steps]['reward'] = e_r
        step_data[total_steps]['total_infections'] = total_infections
        step_data[total_steps]['total_test_num'] = total_test_num
        step_data[total_steps]['total_quara_num'] = total_quara_num
        step_data[total_steps]['score'] = score

    while total_steps < args.max_train_steps:
        # Reset beta_matrix if using beta_change
        if args.use_beta_change:
            env.dynamic_beta_change(type=args.train_beta_change_rule, std=0.1)

        s = env.reset()
        if evaluate_num > START_TRAIN_DL:
            s = generic_predictor.predict(env = env, s = s)  # Predict state
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
            s_, r, done, _ = env.step(a)
            if evaluate_num > START_TRAIN_DL:
                s_ = generic_predictor.predict(env = env, s = s_)  # Predict state

            if args.use_state_norm:
                s_ = state_norm(s_, update=True)

            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            if done and episode_steps != args.max_episode_steps:
                dw = True  # dw is usually False, but GAE uses done
            else:
                dw = False

            # Store data to replay buffer
            for i in range(replay_buffer.env_count):
                replay_buffer.store(episode_steps - 1 + i * args.ODE_period, s[i], a[i], a_logprob[i], r[i], s_[i], dw, done)

            s = s_
            total_steps += args.env_count

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                # RL training: update actor-critic
                a_loss, c_loss, entropy, lr_now = agent.update(replay_buffer, total_steps)
                # DL training: update predictor
                if args.train_dl_in_train:
                    generic_predictor.train(env)

                replay_buffer.count = 0

                writer.add_scalar('train_loss/a_loss', a_loss.item(), global_step=total_steps)
                writer.add_scalar('train_loss/c_loss', c_loss.item(), global_step=total_steps)
                writer.add_scalar('train_loss/entropy', entropy.item(), global_step=total_steps)
                writer.add_scalar('train_loss/lr', lr_now, global_step=total_steps)

                # # Store training loss data by step
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
                generic_predictor.save(evaluate_num + 1)  # Save DL model

                # Store evaluation data by step
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
    args.cost_ratio_test_to_quarantine = 1 / 8  # Test cost to quarantine cost ratio
    rw0 = 500
    rw2 = 1
    rw1 = rw2 * args.cost_ratio_test_to_quarantine / (1/args.ODE_sigma + 1/args.ODE_gamma)  # 1/80
    args.reward_weights = [rw0 * 1000, rw1 * 1000, rw2 * 1000]

    # args.reward_weights = [1000, 1/8/7, 1]

    # Reduce lamda for more exploration
    args.lamda = 0.9
    args.entropy_coef = 0.2

    # args.use_adv_norm = True
    args.use_state_norm = True

    # args.use_reward_shaping = True

    # Predictor config
    args.predictor_type = 'none'
    args.train_dl_in_train = False
    args.train_dl_in_eval = False

    # Delayed quarantine
    args.use_delay_quara = True

    # Reward function response distribution (enabled by default)
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
    # # Save to file
    # import pickle
    # with open('draw/infection_and_action_map/env.pkl', 'wb') as f:  # 'wb' means open in binary write mode
    #     pickle.dump(env, f)

    # Test in uncertain environment
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
    """Compare reward functions with/without response distribution mechanism."""
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

        # Test in uncertain environment
        args.use_obs_imperfect = True
        args.obs_imperfect_down = 0.1
        args.obs_imperfect_up = 1
        _, info = my_test(args, env_count=100)
        info['name'] = 'uncertain_' + ("without_response_distribution" if drd else "with_response_distribution")
        infos.append(info)
    # Export table
    processed_data = [
        {
            "name": info["name"],
            "ep_r": info["ep_r"].mean().item(),
            "total_infections": info["total_infections"].mean().item(),
            "total_test_num": info["total_test_num"].mean().item(),
            "total_quara_num": info["total_quara_num"].mean().item(),
            "total_control_cost": info["total_control_cost"].mean().item(),
            "score": info["score"].mean().item(),  # Calculate mean and convert to Python scalar
            "score_avg": info["score_avg"].item(),
            "avg_zone_score": info["avg_zone_score"].mean().item(),
        }
        for info in infos
    ]

    # Generate DataFrame
    df = pd.DataFrame(processed_data)

    csv_file_path = 'draw/ablation_responsibility_distribution' + '/' + f'{args.R0}_control_result.csv'
    df.to_csv(csv_file_path, index=False)
    print("DataFrame has been exported to CSV: ", csv_file_path)

def compare_different_reward_func_plot():
    """Compare reward functions with/without response distribution, run 10 random seeds."""
    # 1. Generate 10 random seeds
    np.random.seed(42)  # Fixed seed for reproducibility
    seeds = np.random.randint(1000, 9999, size=10)
    print(f"Random seeds: {seeds}")

    # 2. Initialize data collection: store reward curves
    # Key format: (disable_response_distribution, seed)
    reward_data = {}

    # 3. Run all experiments
    args.eval_before_train_start = True
    for seed in seeds:
        print(f"\n===== Starting seed {seed} =====")
        for drd in [True, False]:  # True: without response distribution; False: with response distribution
            # Set current experiment params
            args.disable_response_distribution = drd
            args.use_obs_imperfect = False
            args.daily_imported_cases = 1
            args.max_train_steps = 2400 * 20
            args.show_fig = False  # Disable plots during training

            # Run main training function
            print(f"--- Response distribution: {'disabled' if drd else 'enabled'} ---")
            pd_data = main(args, seed=seed)  # Returns DataFrame with step and reward

            # Extract and store reward curve
            if 'step' in pd_data.columns and 'reward' in pd_data.columns:
                reward_curve = pd_data[['step', 'reward']].copy()
                reward_curve['seed'] = seed
                reward_curve['drd'] = 'without_RD' if drd else 'with_RD'
                reward_data[(drd, seed)] = reward_curve
            else:
                print(f"Warning: seed {seed} (drd={drd}) returned invalid data")

    # 4. Save data
    all_rewards = pd.concat(reward_data.values(), ignore_index=True)
    save_dir = "draw/ablation_responsibility_distribution"
    # Create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"reward_data_{args.R0}_{timestamp}.csv")
    all_rewards.to_csv(save_path, index=False)
    print(f"Data saved to: {save_path}")



if __name__ == '__main__':


    _config_args()

    # Low R0 experiment
    # args.ODE_beta = 0.4
    # args.R0 = 'low'

    # Community environment training
    train_in_community()

    # Uncertainty environment training
    # train_in_uncertainty()

    # Compare reward functions with/without response distribution
    # for R0 in ['low', 'high']:
    #     args.R0 = R0
    #     args.ODE_beta = 0.8 if R0 == 'high' else 0.4
    #
    #     # compare_different_reward_func()
    #     compare_different_reward_func_plot()
