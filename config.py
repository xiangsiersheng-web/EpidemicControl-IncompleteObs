import numpy as np
import argparse

parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
#A_description
parser.add_argument("--A_description", type=str, default="训练场景描述", help="Algorithm description")
parser.add_argument("--device_name", type=str, default='cuda:0', help="device_name")


parser.add_argument("--max_train_steps", type=int, default=int(1.2e6), help=" Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=1.2e4,
                    help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--save_freq", type=int, default=10, help="Save frequency")
parser.add_argument("--batch_size", type=int, default=2400, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")

parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=3, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.2, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
parser.add_argument("--use_reward_shaping", type=bool, default=False, help="use_reward_shaping")

parser.add_argument("--hidden_width", type=int, default=256,
                    help="The number of neurons in hidden layers of the neural network")

parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")

parser.add_argument("--load_model", type=bool, default=False, help="load pretrained model or Not")
parser.add_argument("--model_idx", type=int, default=int(100), help="which model to load")
parser.add_argument("--experiment_idx", type=int, default=int(4),
                    help="experiment id determining the reward function and save path")


# 训练有关参数
parser.add_argument("--simulate_scale", type=str, default='district', help="simulate_scale, district or community")
parser.add_argument("--local_obs_dim", type=int, default=int(2), help="local state dimension, I R delta_I (action)")
parser.add_argument("--action_dim", type=int, default=int(10), help="action dimension")
parser.add_argument("--zone_num", type=int, default=74, help="zone_num")
parser.add_argument("--WINDOW_SIZE", type=int, default=7, help="window_size")
parser.add_argument("--use_asymmetric", type=bool, default=False, help="use_asymmetric")
parser.add_argument("--state_standard_scale", type=int, default=1e4, help="state_normalized_scale")
parser.add_argument("--use_bc_init", type=bool, default=False, help="use_bc_init")
# 状态中是否包含beta,action,toJ
parser.add_argument("--state_contain_beta", type=bool, default=False, help="state_include_beta")
parser.add_argument("--state_contain_action", type=bool, default=False, help="state_include_action")
parser.add_argument("--state_contain_toJ", type=bool, default=False, help="state_include_toJ")
# actor critic 的网络架构
parser.add_argument("--actor_critic_model", type=str, default="mlp", help="actor_critic_model")

# 环境初始化参数
parser.add_argument("--city", type=str, default='sz', help="case")
parser.add_argument("--R0", type=str, default='high', help="scenario")
parser.add_argument("--reward_mode", type=int, default=4, help="reward_mode")
# ODE 有关参数
parser.add_argument("--ODE_Pm", type=float, default=0.8, help="ODE_Pm")
parser.add_argument("--ODE_beta", type=float, default=0.8, help="ODE_beta")
parser.add_argument("--ODE_sigma", type=float, default=1 / 3, help="ODE_sigma")
parser.add_argument("--ODE_gamma", type=float, default=1 / 7, help="ODE_gamma")
parser.add_argument("--ODE_period", type=int, default=120, help="ODE_period")
parser.add_argument("--ODE_zero_threshold", type=float, default=0.0, help="ODE_zero_threshold")
# 检测资源效率函数
parser.add_argument("--detection_efficiency_exp_param", type=float, default=0.6, help="检测资源效率函数的指数参数")
# 是否绘图显示
parser.add_argument("--show_fig", type=bool, default=False, help="show_fig")

"""环境不确定性有关参数"""

parser.add_argument("--eval_before_train_start", type=bool, default=False, help="是否在训练前进行一次评估，除奖励函数消融实验情况下，别的情形不启用")

parser.add_argument("--rl_type", type=str, default="certainty", help="标注RL是否要在非确定环境下训练")

## 不完全观测
parser.add_argument("--I_obs_imperfect", type=bool, default=False, help="I_obs_imperfect")
parser.add_argument("--use_obs_imperfect", type=bool, default=False, help="use_obs_imperfect")
# obs_imperfect_up
parser.add_argument("--obs_imperfect_up", type=float, default=1.0, help="obs_imperfect_up")
parser.add_argument("--obs_imperfect_down", type=float, default=0.1, help="obs_imperfect_down")

## 环境非稳态
parser.add_argument("--gather_to_some_region", type=bool, default=False, help="gather_to_some_region")

## 状态转移不确定
# 传播参数的变化
parser.add_argument("--use_beta_change", type=bool, default=False, help="use_beta_change")
parser.add_argument("--env_beta_change_rule", type=str, default="none", help="Defines the rule for changing beta over time or space")
parser.add_argument("--train_beta_change_rule", type=str, default="none", help="Defines the rule for changing beta over time or space")
parser.add_argument("--test_beta_change_rule", type=str, default="none", help="Defines the rule for changing beta over time or space")

# 动作的不确定
parser.add_argument("--use_action_uncertainty", type=bool, default=False, help="use_action_uncertainty")

try:
    args = parser.parse_args()
except SystemExit as e:
    print("命令行参数解析失败，使用默认参数。")
    args = parser.parse_args(args=[])