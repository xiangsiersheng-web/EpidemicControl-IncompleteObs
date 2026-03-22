import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import os

from uncertainty.obs_imperfect.main_rebuild import _get_history_padding
from utils.normalization import Normalization, RewardScaling
from algorithm.ppo_discrete_gpu import PPO_discrete_gpu
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
# from uncertainty.obs_imperfect.gru_model import RebuildGruModel, RebuildGru
from utils.general_functions import mask_params_to_str


def _evaluate_policy(args, env, agent, state_norm=None, evaluate_num=0, rebuild_method='ode_formula'):
    """
    Evaluate model: start from initial state, complete full episode trajectories
    """

    # If using beta_change, need to reset beta_matrix
    if args.use_beta_change:
        env.dynamic_beta_change(type=args.test_beta_change_rule, std=0.2)
    s = env.reset()
    POP = env.POP
    OD = env.OD[0].unsqueeze(0).expand(env.env_count, -1, -1)
    OD_t = OD.transpose(1,2)
    done = False

    rebuild_states1 = torch.zeros((env.env_count, env.period + 1, env.ZONE_NUM, 2), device=env.device)  # For recording
    rebuild_states2 = torch.zeros((env.env_count, env.period + 1, env.ZONE_NUM, 2), device=env.device)  # For agent, uses actual observations before day 5
    start_rebuild_day = 1 if draw_data_output_path is None or draw_data_output_path != "../../draw/map_comparison" else 8

    # Define two reconstruction evaluation metrics
    predict_slice_effect = -1.0
    predict_total_effect = -1.0

    # Load distance matrix for IDW
    # Load distance matrix
    matrix_path = "../../data/sz/community_654/distance_matrix.npy"
    distance_matrix = np.load(matrix_path)
    distance_matrix = torch.from_numpy(distance_matrix).float().to(env.device)

    while not done:
        if args.use_rebuild:
            history_action = _get_history_padding(env.actions, env.day - 1, 7)
            obs = _get_history_padding(env.history_local_obs[:, :, :, :2], env.day, 7) * env.POP.unsqueeze(
                1).unsqueeze(-1)

            # a. Get undetected regions
            u_p_test, u_p_quara = env._action_to_u(env.actions[:, env.day - 2, :])
            # obs_mask = obs[:, -1, :, 0] == 0
            obs_mask = u_p_test == 0

            """Perform prediction, i.e., information reconstruction"""
            if rebuild_method == 'ode_formula':
                """Reconstruction logic 1: use mathematical formula"""
                predict = _rebuild_ode_formula(OD_t=OD_t, OD=OD, POP=POP, obs=obs, obs_mask=obs_mask)
            elif rebuild_method == 'gnn_gru':
                """Reconstruction logic 2: use args.trainer"""
                predict = args.trainer.predict(obs=obs, actions=history_action) # (env.env_count, env.ZONE_NUM, 2)
            elif rebuild_method == 'idw':
                """Reconstruction logic 3: inverse distance weighting"""
                predict = _rebuild_idw(obs=obs, obs_mask=obs_mask, distance_matrix=distance_matrix)
            else:
                raise ValueError("rebuild_method error. rebuild_method = ", rebuild_method)
            """Post-reconstruction processing"""
            # Method 1: directly use known current cases to predict unknown current cases
            rebuild_states1[:, env.day - 1, :, 0] = predict[:, :, 0]    # Record estimated current cases
            rebuild_states1[:, env.day - 1, :, 1] = predict[:, :, 1]    # Record estimated new cases
            env.record_rebuild_state(rebuild_states1[:, env.day - 1, :, :].clone())

            # # Method 2: use estimated new cases to estimate unknown current cases (yesterday's remaining + today's new)
            # rebuild_states1[:, env.day - 1, :, 0] = rebuild_states1[:, env.day - 2, :, 0] * (1 - 1 / (3)) \
            #                                         + rebuild_states1[:, env.day - 1, :, 1]
            # rebuild_states1[:, env.day - 1, :, 0][~obs_mask] = obs[:, -1, :, 0][~obs_mask] # For detected regions, use actual observations

            # Correlation coefficient calculation
            # x = env.simState[:, :,[env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].mean(dim=0).sum(dim=-1)[obs_mask[0]]
            # y = rebuild_states1[:, env.day - 1, :, 0].mean(dim=0)[obs_mask[0]]
            # x2 = env.daily_new_E[:, env.day - 1, :].mean(dim=0)[obs_mask[0]]
            # y2 = rebuild_states1[:, env.day - 1, :, 1].mean(dim=0)[obs_mask[0]]
            x = env.simState[:, :,
                [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].mean(dim=0).sum(
                dim=-1)
            y = rebuild_states1[:, env.day - 1, :, 0].mean(dim=0)
            x2 = env.daily_new_E[:, env.day - 1, :].mean(dim=0)
            y2 = rebuild_states1[:, env.day - 1, :, 1].mean(dim=0)
            if env.day - 1 == 2: # Record prediction effect on day 2
                predict_slice_effect = _evaluate_predict_effects(x2, y2)
            if env.day in range(2,10):
                _analysis_rebuild_effect(x, y, title=f"Day {env.day - 1} (curr_EI)")
                _analysis_rebuild_effect(x2, y2, title=f"Day {env.day - 1} (Eun)")

            if env.day >= start_rebuild_day:
                """Use reconstructed information for decision-making starting from day 5"""
                rebuild_states2[:, env.day - 1, :, :] = rebuild_states1[:, env.day - 1, :, :]
            else:
                rebuild_states2[:, env.day - 1, :, :] = obs[:, -1, :, :]

            r_s = _get_history_padding(rebuild_states2 / POP.unsqueeze(1).unsqueeze(-1), env.day, env.WINDOW_SIZE)
            r_s = torch.cat((r_s, s[:,:,:,2:]), dim=-1)
            s = r_s
        if args.use_state_norm:
            s = state_norm(s, update=False)
        a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
        # if env.day < start_rebuild_day and draw_data_output_path == "../../draw/map_comparison":
        #     a = torch.zeros_like(a)
        #     no_mask = a[0, :] == 0
        #     for region_idx in range(0, env.ZONE_NUM):
        #         if region_idx % 10 == 1:
        #             no_mask[region_idx] = True
        #         else:
        #             no_mask[region_idx] = False
        #     a[:, no_mask] = 3
        s_, r, done, info = env.step(a)

        s = s_

    if args.show_fig:
        env.render(title="eval:" + str(evaluate_num))
        plot_all_env(rebuild_states1, env)

        # for region_idx in range(0, 20):
        #     plot_one_region(rebuild_states, env, 0, region_idx)

    # Evaluate overall prediction effect TODO: Current RMSE calculation averages across environments first, should it be changed to calculate RMSE per environment then average?
    predict_total = rebuild_states1[:, :, :, 0].mean(dim=0).sum(dim=-1)
    target_total = (env.simRes[:, :, :,[env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]]
                    .sum(dim=[-2, -1]).mean(dim=0))
    predict_total_effect = _evaluate_predict_effects(predict_total, target_total)

    if not args.use_rebuild:
        # If no information reconstruction, use observations and targets for evaluation
        predict_slice = (env.history_local_obs[:, 2, :, 1] * env.POP).mean(dim=0)
        total_slice = (env.daily_new_E[:, 2, :].mean(dim=0))
        predict_slice_effect = _evaluate_predict_effects(predict_slice, total_slice)

        predict_total = (env.history_local_obs[:, :, :, 0] * env.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)
        predict_total_effect = _evaluate_predict_effects(predict_total, target_total)

    info["predict_slice_effect"] = predict_slice_effect
    info["predict_total_effect"] = predict_total_effect

    info["missing_rate"] = env.get_observation_missing_rate()

    return info

def _evaluate_predict_effects(predict: torch.Tensor, target: torch.Tensor) -> float:
    # Use RMSE to evaluate prediction effect
    # Calculate mean squared error (MSE), then take square root to get RMSE
    mse = torch.mean((predict - target) ** 2)
    rmse = torch.sqrt(mse)

    # Return Python float (not tensor)
    return rmse.item()

def _rebuild_ode_formula(OD_t, OD, POP, obs, obs_mask):
    # b. Formula reconstruction logic
    predict = torch.bmm(OD_t, obs[:, -1, :, :]) / torch.bmm(OD_t, POP.unsqueeze(-1))  # (env.env_count, env.ZONE_NUM, 2)
    predict = torch.bmm(OD, predict) * POP.unsqueeze(-1)
    predict = torch.clip(predict, min=0)
    # c. Numerical correction
    numerical_ratio = [(obs[i, -1, ~obs_mask[i], :]).sum(dim=0) / (predict[i, ~obs_mask[i], :].sum(dim=0) + 1e-16)
                       for i in range(obs.shape[0])]
    numerical_ratio = torch.stack(numerical_ratio).unsqueeze(1)
    predict = predict * numerical_ratio
    predict[~obs_mask] = obs[:, -1, :, :][~obs_mask]  # For regions with observations, fill back with actual observed values.
    predict = torch.max(predict, obs[:, -1, :, :])  # Take the larger of reconstruction and observation, should this be kept?

    return predict



def _rebuild_idw(obs, obs_mask, distance_matrix, power=2.0, eps=1e-8):
    """
        Vectorized IDW algorithm to fill missing values (regions where obs_mask=True)
    """
    obs = obs[:, -1, :, :].clone()
    predict = obs.clone()

    # Process each batch and each feature independently
    for batch in range(predict.shape[0]):
        for feature in range(predict.shape[2]):
            current_obs = obs[batch, :, feature]

            current_mask = obs_mask[batch, :]

            missing_indices = torch.where(current_mask)[0]

            known_indices = torch.where(~current_mask)[0]

            if len(known_indices) == 0:
                continue

            # --- Vectorized core computation ---
            # Extract distances between missing points and known points (n_missing, n_known)
            dists = distance_matrix[missing_indices][:, known_indices]  # (n_missing, n_known)

            # Calculate weights (n_missing, n_known)
            weights = 1.0 / (dists.pow(power) + eps)

            # Known values (n_known,)
            known_values = obs[batch, known_indices, feature]

            # Weighted sum (n_missing,)
            weighted_sum = (weights * known_values).sum(dim=1)
            sum_weights = weights.sum(dim=1)

            # Fill results
            predict[batch, missing_indices, feature] = weighted_sum / sum_weights


    return predict


def _analysis_rebuild_effect(x, y, title ="Day 0 (curr_EI)"):
    """x is the true value, y is the predicted value"""
    if len(x) < 2 or y.sum() == 0 or x.sum() == 0:
        r, p_value = 0, 0
        r2 = 0
    else:
        r, p_value = pearsonr(x.cpu().numpy(), y.cpu().numpy())
        r2 = r2_score(x.cpu().numpy(), y.cpu().numpy())
        # # Plot x y scatter plot
        # plt.plot(x.cpu().numpy(), y.cpu().numpy(), 'o', color='blue')
        # plt.xlabel("True state")
        # plt.ylabel("Rebuild state")
        # plt.title(f"{title} (pearson:{r:.4f}, R2:{r2:.4f})")
        # plt.show()
    print("==============", title, "==============")
    print(len(x))
    print(f"Pearson Correlation Coefficient: {r:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"R2: {r2:.4f}")


def plot_all_env(rebuild_states, env):

    # Comparison 1
    first_vector_1 = rebuild_states[:, :, :, 0].sum(dim=2).mean(dim=0)  # shape (120,)
    first_vector_2 = env.simRes[:, :, :,
                     [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(
        dim=[-2, -1]).mean(dim=0)
    first_vector_3 = (env.history_local_obs[:, :, :, 0] * env.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)

    # Comparison 2
    second_vector_1 = rebuild_states[:, :, :, 1].sum(dim=2).mean(dim=0)  # shape (120,)
    # second_vector_2 = env.simRes[:, :, :, env.E_undetected].sum(dim=-1).mean(dim=0)  # shape (120,) todo:should this be replaced
    second_vector_2 = env.daily_new_E.sum(dim=-1).mean(dim=0)  # shape (120,) This is the prediction target for new cases
    second_vector_3 = (env.history_local_obs[:, :, :, 1] * env.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)

    _plot_6_compare(first_vector_1, first_vector_2, first_vector_3, second_vector_1, second_vector_2, second_vector_3)


def _plot_6_compare(first_vector_1, first_vector_2, first_vector_3, second_vector_1, second_vector_2, second_vector_3, region_idx=None):
    # Plot figure
    plt.figure(figsize=(10, 8))
    # First comparison (first pair of curves)
    plt.subplot(2, 1, 1)
    plt.plot(first_vector_1.cpu().numpy(), label="Rebuild States - E/I", color='blue')
    plt.plot(first_vector_2.cpu().numpy(), label="True - E/I", color='green')
    plt.plot(first_vector_3.cpu().numpy(), label="Obs - E/I", color='orange')
    title1 = "Comparison 1: Rebuild States vs SimRes"
    if region_idx is not None:
        title1 += f" (Region {region_idx})"
    plt.title(title1)
    plt.xlabel("Index (Time Step)")
    plt.ylabel("Value")
    plt.legend()
    # Second comparison (second pair of curves)
    plt.subplot(2, 1, 2)
    plt.plot(second_vector_1.cpu().numpy(), label="Rebuild States - E/I", color='blue')
    plt.plot(second_vector_2.cpu().numpy(), label="True New EI", color='green')
    plt.plot(second_vector_3.cpu().numpy(), label="Obs New EI", color='orange')
    # plt.plot(second_vector_4.cpu().numpy(), label="Daily Detect EI", color='blue')
    title2 = "Comparison 2: Rebuild States vs Daily New EI"
    if region_idx is not None:
        title2 += f" (Region {region_idx})"
    plt.title(title2)
    plt.xlabel("Index (Time Step)")
    plt.ylabel("Value")
    plt.legend()
    # Display figure
    plt.tight_layout()
    plt.show()

def _extract_state_from_env(env: EpidemicModel, path="../../draw/monitoring_warning_competition/perfect_report"):
    # Extract self.simRes = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 9), device=self.device) from env
    simRes = env.simRes.cpu().numpy()
    np.save(path + '/simRes.npy', simRes)
    print("simRes shape: ", simRes.shape)

    # Extract new cases env.daily_new_E
    daily_new_E = env.daily_new_E.cpu().numpy()
    np.save(path + '/daily_new_E.npy', daily_new_E)
    print("daily_new_E shape: ", daily_new_E.shape)

    # Extract obs
    obs = env.extract_observe().cpu().numpy()
    np.save(path + '/obs.npy', obs)
    print("obs shape: ", obs.shape)

    # Extract actions:
    actions = env.actions.cpu().numpy()
    np.save(path + '/actions.npy', actions)
    print("actions shape: ", actions.shape)

    # Extract daily quarantine count
    daily_quara_num = env.extract_daily_quara_num().cpu().numpy()
    np.save(path + '/daily_quara_num.npy', daily_quara_num)
    print("daily_quara_num shape: ", daily_quara_num.shape)

    # Extract rebuild_state
    rebuild_states = env.extract_rebuild_state().cpu().numpy()
    np.save(path + '/rebuild_states.npy', rebuild_states)
    print("rebuild_states shape: ", rebuild_states.shape)


def main():
    from train_gpu import _config_args
    args = _config_args()
    # args.device_name = 'cpu'
    args.daily_imported_cases = 1
    args.show_fig = True

    # Low R0 experiment
    args.ODE_beta = 0.4
    args.R0 = 'low'

    # Validate the effect of detection resource efficiency function exponent changes
    args.detection_efficiency_exp_param = 0.65

    # Specify agent to load
    args.max_train_steps = 2400 * 20
    args.model_idx = 18
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent_original_dir = agent.directory
    agent.load(args.model_idx)
    # State normalization
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # Initialize a perfect observation environment
    args.env_data_dir = '../../data/'
    env_obs_perfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_perfect.seed()
    # Do not use information reconstruction
    args.use_rebuild = False
    infos = []
    info_certain = _evaluate_policy(args, env_obs_perfect, agent, state_norm=state_norm)
    info_certain['name'] = 'Certain'
    infos.append(info_certain)

    # Initialize an imperfect observation environment
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    # Do not use information reconstruction
    args.use_rebuild = False
    info_uncertain = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm)
    info_uncertain['name'] = 'Uncertain'
    infos.append(info_uncertain)

    # Use information reconstruction
    args.use_rebuild = True

    # 1. Mechanism
    # 1.1. ODE
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='ode_formula')
    info_rebuild['name'] = 'ode_formula'
    infos.append(info_rebuild)

    # 2. Data-driven
    # 2.1.idw
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='idw')
    info_rebuild['name'] = 'idw'
    infos.append(info_rebuild)

    # 2.2. gnn-gru
    # Load model
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2
    # Initialize an environment to use its OD POP data and action_to_u method
    env_temp = EpidemicModel(args, env_count=1)
    env_temp.POP = None
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_ordinary.pth'))
    args.trainer = trainer
    # Test
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    info_rebuild['name'] = 'gnn_gru_ordinary'
    infos.append(info_rebuild)

    # 3. Mechanism + data-driven
    # 3.1. Agent dataset
    # Load model
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2
    # Initialize an environment to use its OD POP data and action_to_u method
    env_temp = EpidemicModel(args, env_count=1)
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_agent.pth'))
    args.trainer = trainer
    # Test
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    info_rebuild['name'] = 'gnn_gru_agent'
    infos.append(info_rebuild)

    # 3.2. Random dataset
    # Load model
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2
    # Initialize an environment to use its OD POP data and action_to_u method
    env_temp = EpidemicModel(args, env_count=1)
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_random.pth'))
    args.trainer = trainer
    # Test
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    info_rebuild['name'] = 'gnn_gru_random'
    infos.append(info_rebuild)

    # 3.3. Include actions
    # Load model
    from uncertainty.obs_imperfect.gru_gnn_model_v2 import RebuildGruGNNModel, RebuildGruGNN
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2
    # Initialize an environment to use its OD POP data and action_to_u method
    env_temp = EpidemicModel(args, env_count=1)
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name, node_output_size=48)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len, env=env_temp)
    trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_with_action.pth'))
    args.trainer = trainer
    # Test
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    info_rebuild['name'] = 'gnn_gru_with_action'
    infos.append(info_rebuild)

    # 4. e2e
    # 4.1. RL trained directly under incomplete observation
    # Load agent
    args.use_rebuild = False
    args.rl_type = 'uncertainty'
    args.predictor_type = 'none'
    args.use_obs_imperfect = True  # Indicates loading agent co-trained in uncertain environment
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent.load(args.model_idx)
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')
    # Test
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm)
    info_rebuild['name'] = 'rl_train_in_imperfect_obs'
    infos.append(info_rebuild)

    # 4.2. e2e
    args.use_rebuild = True
    for p_type in ['rebuild_gru_gnn_model_with_action', 'rebuild_gru_gnn_model_no_action']:
        # Load agent
        args.model_idx = 19 if args.R0 == 'high' else 18
        args.rl_type = 'uncertainty'
        args.predictor_type = p_type
        args.use_obs_imperfect = True # Indicates loading agent co-trained in uncertain environment
        agent = PPO_discrete_gpu(args)
        agent.directory = '../../' + agent.directory
        agent.load(args.model_idx)
        if args.use_state_norm:
            state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
            state_norm.load(agent.directory, filename='state_norm.pth')
        # Initialize model gnn-gru-with-action-edge2edge (agent stateNormal both need to be reloaded)
        from uncertainty.obs_imperfect.generic_predictor import GenericPredictor
        args.env_count = repeats
        predictor = GenericPredictor(args, agent.directory)
        predictor.load(args.model_idx)
        args.trainer = predictor.get_trainer()

        # Test
        args.use_obs_imperfect = True
        env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
        env_obs_imperfect.seed()
        info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
        info_rebuild['name'] = 'gnn_gru_e2e' + p_type
        infos.append(info_rebuild)


    # Export data table
    import pandas as pd

    # Preprocessing: calculate specific fields for each dictionary
    processed_data = [
        {
            "name": info["name"],
            "predict_total_effect": info["predict_total_effect"],
            "predict_slice_effect": info["predict_slice_effect"],
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

    # Output to Excel
    excel_path = agent_original_dir + '/' + f'processed_scores_{repeats}_detection_{args.detection_efficiency_exp_param}.xlsx'
    csv_file_path = agent_original_dir + '/' + f'processed_scores_{repeats}_detection_{args.detection_efficiency_exp_param}.csv'
    df.to_csv(csv_file_path, index=False)
    print("DataFrame has been exported to CSV: ", csv_file_path)
    df.to_excel(excel_path, index=False)
    print("DataFrame has been exported to Excel: ", excel_path)

def extract_data_for_partial_day():
    """
    Extract data from env for plotting

    Plot types:
        Map comparison: Use ODE mechanism reconstruction as "rebuild_info" source, better control to start reconstruction from day 7/8
        Monitoring warning competition: Focus on time display, so use `3.1. agent dataset` as "rebuild_info" source
    """
    # draw_data_output_path = "../../draw/map_comparison"
    draw_data_output_path = "../../draw/monitoring_warning_competition"
    repeats = 2

    from train_gpu import _config_args
    args = _config_args()
    args.daily_imported_cases = 1
    args.show_fig = True

    # Specify agent to load
    args.max_train_steps = 2400 * 20
    args.model_idx = 18
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent_original_dir = agent.directory
    agent.load(args.model_idx)
    # State normalization
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # Initialize a perfect observation environment
    args.env_data_dir = '../../data/'
    env_obs_perfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_perfect.seed()
    # Do not use information reconstruction
    args.use_rebuild = False
    infos = []
    info_certain = _evaluate_policy(args, env_obs_perfect, agent, state_norm=state_norm)
    info_certain['name'] = 'Certain'
    infos.append(info_certain)
    _extract_state_from_env(env_obs_perfect, draw_data_output_path + "/perfect_report")

    # Initialize an imperfect observation environment
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    # Set mask
    report_mask = torch.zeros((env_obs_imperfect.env_count, env_obs_imperfect.ZONE_NUM), dtype=torch.bool, device=env_obs_imperfect.device)
    for i in range(env_obs_imperfect.env_count):
        report_idx = []
        for region_idx in range(env_obs_imperfect.ZONE_NUM):
            if region_idx % 3 == 0:
                report_idx.append(region_idx)

        report_mask[i, report_idx] = True
    env_obs_imperfect.set_mask(~report_mask)
    # Do not use information reconstruction
    args.use_rebuild = False
    info_uncertain = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm)
    info_uncertain['name'] = 'Uncertain'
    infos.append(info_uncertain)
    _extract_state_from_env(env_obs_imperfect, draw_data_output_path + "/partial_observable")

    # Use information reconstruction
    args.use_rebuild = True

    # 1. Mechanism
    # 1.1. ODE
    args.use_obs_imperfect = True
    env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    env_obs_imperfect.seed()
    env_obs_imperfect.set_mask(~report_mask)
    info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='ode_formula')
    info_rebuild['name'] = 'ode_formula'
    infos.append(info_rebuild)
    _extract_state_from_env(env_obs_imperfect, draw_data_output_path + "/rebuild_info")

    # # 3.1. agent dataset
    # # Load model
    # from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
    # seq_len = 7
    # hidden_size = 128
    # output_size = 2
    # num_layers = 2
    # # Initialize an environment to use its OD POP data and action_to_u method
    # env_temp = EpidemicModel(args, env_count=1)
    # model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
    #                            env=env_temp, device_name=args.device_name)
    # trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    # trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_agent.pth'))
    # args.trainer = trainer
    # # Test
    # args.use_obs_imperfect = True
    # env_obs_imperfect = EpidemicModel(args, env_count=repeats, is_evaluation=True)
    # env_obs_imperfect.seed()
    # env_obs_imperfect.set_mask(~report_mask)
    # info_rebuild = evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
    # info_rebuild['name'] = 'gnn_gru_agent'
    # infos.append(info_rebuild)
    # extract_state_from_env(env_obs_imperfect, draw_data_output_path + "/rebuild_info")

def _analysis_observations_missing_rate(args, missing_rate_table_file_path=""):
    """
    Analyze the relationship between observation missing rate and score
    """
    repeats = 20

    # Specify agent to load
    args.max_train_steps = 2400 * 20
    args.model_idx = 18
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    agent_original_dir = agent.directory
    agent.load(args.model_idx)
    # State normalization
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # Use a table to record results for later plotting, column names: mask_rate_up, mask_rate_down, mask_duration_up, mask_duration_down, name, missing_rate, score
    rows = []

    random_report_rate_down = random.Random(3047)
    seeds = [int(random_report_rate_down.uniform(10, 10000)) for _ in range(repeats)]
    for mask_rate_down in [0, 0.2, 0.4, 0.6, 0.8]:
        mask_rate_up = mask_rate_down + 0.2
        args.mask_rate_up = mask_rate_up
        args.mask_rate_down = mask_rate_down
        for mask_duration_down in [14, 28, 42, 56, 70, 84, 98]:
            mask_duration_up = mask_duration_down + 14
            args.mask_duration_up = mask_duration_up
            args.mask_duration_down = mask_duration_down
            mask_str = mask_params_to_str(args)

            # for seed in [3047, 42, 69, 420, 666]:
            for seed in seeds:
                # Initialize a perfect observation environment
                args.env_data_dir = '../../data/'
                args.use_obs_imperfect = False
                env_obs_perfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_perfect.seed(seed)
                # Do not use information reconstruction
                args.use_rebuild = False
                infos = []
                info_certain = _evaluate_policy(args, env_obs_perfect, agent, state_norm=state_norm)
                info_certain['name'] = 'Certain'
                infos.append(info_certain)

                # Initialize an imperfect observation environment
                args.use_obs_imperfect = True
                args.obs_imperfect_down = 0.1
                args.obs_imperfect_up = 1
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                # Do not use information reconstruction
                args.use_rebuild = False
                info_uncertain = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm)
                info_uncertain['name'] = 'Uncertain'
                infos.append(info_uncertain)

                # Use information reconstruction
                args.use_rebuild = True

                # 1. Mechanism
                # 1.1. ODE
                args.use_obs_imperfect = True
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='ode_formula')
                info_rebuild['name'] = 'ode_formula'
                infos.append(info_rebuild)

                # 2. Data-driven
                # 2.1.idw
                args.use_obs_imperfect = True
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='idw')
                info_rebuild['name'] = 'idw'
                infos.append(info_rebuild)

                # 2.2. gnn-gru
                # Load model
                from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
                seq_len = 7
                hidden_size = 128
                output_size = 2
                num_layers = 2
                # Initialize a temporary environment to use its OD POP data and action_to_u method
                env_temp = EpidemicModel(args, env_count=1)
                env_temp.POP = None
                model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                                           env=env_temp, device_name=args.device_name)
                trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
                trainer.load(os.path.join(agent.directory, f'rebuild_gru_gnn_model_v1_ordinary.pth'))
                args.trainer = trainer
                # Test
                args.use_obs_imperfect = True
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm, rebuild_method='gnn_gru')
                info_rebuild['name'] = 'gnn_gru_ordinary'
                infos.append(info_rebuild)

                # 3.1. agent dataset
                # Load model
                from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
                seq_len = 7
                hidden_size = 128
                output_size = 2
                num_layers = 2
                # Initialize an environment to use its OD POP data and action_to_u method
                env_temp = EpidemicModel(args, env_count=1)
                model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size,
                                           gru_num_layers=num_layers,
                                           env=env_temp, device_name=args.device_name)
                trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
                trainer.load(os.path.join(agent.directory, 'rebuild_gru_gnn_model_v1_agent.pth'))
                args.trainer = trainer
                # Test
                args.use_obs_imperfect = True
                env_obs_imperfect = EpidemicModel(args, env_count=1, is_evaluation=True)
                env_obs_imperfect.seed(seed)
                info_rebuild = _evaluate_policy(args, env_obs_imperfect, agent, state_norm=state_norm,
                                                rebuild_method='gnn_gru')
                info_rebuild['name'] = 'gnn_gru_agent'
                infos.append(info_rebuild)


                missing_rate_avg = info_rebuild['missing_rate'].mean().item()
                print(f"mask_rate_down: {mask_rate_down:.2f}, "
                      f"mask_rate_up: {mask_rate_up:.2f}, "
                      f"mask_duration_down: {mask_duration_down:.2f}, "
                      f"mask_duration_up: {mask_duration_up:.2f}, "
                      f"missing_rate_avg: {missing_rate_avg:.4f}")

                # Extract missing_rate and score correspondence for each name from infos
                for info in infos:
                    for i in range(info['missing_rate'].shape[0]):
                        missing_rate = info["missing_rate"][i].item()
                        if info['name'] == 'Certain':
                            missing_rate = info_uncertain["missing_rate"][i].item()
                        row = {
                            "mask_rate_up": float(args.mask_rate_up),
                            "mask_rate_down": float(args.mask_rate_down),
                            "mask_duration_up": int(args.mask_duration_up),
                            "mask_duration_down": int(args.mask_duration_down),
                            "name": info["name"],
                            "missing_rate": missing_rate,
                            "score": info["score"][i].item(),
                            "predict_total_effect": info["predict_total_effect"],
                        }
                        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_excel(missing_rate_table_file_path, index=False)

def analysis_observations_missing_rate():
    from train_gpu import _config_args

    args = _config_args()
    # args.device_name = 'cpu'
    args.daily_imported_cases = 1
    args.show_fig = False

    missing_rate_table_file_path = f"../../draw/missing_rate_analysis/missing_rate_table_{args.R0}.xlsx"
    _analysis_observations_missing_rate(args, missing_rate_table_file_path)

    # Low R0 experiment
    args.ODE_beta = 0.4
    args.R0 = 'low'
    missing_rate_table_file_path = f"../../draw/missing_rate_analysis/missing_rate_table_{args.R0}.xlsx"
    _analysis_observations_missing_rate(args, missing_rate_table_file_path)


repeats = 100
draw_data_output_path = None
if __name__ == '__main__':
    main()
    # extract_data_for_partial_day()

    # analysis_observations_missing_rate()

