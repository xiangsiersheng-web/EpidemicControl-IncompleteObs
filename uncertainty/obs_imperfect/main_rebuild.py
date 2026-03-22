import glob
import os

import numpy as np

from algorithm.ppo_discrete_gpu import PPO_discrete_gpu
from environment.uncertain_seir_vector_v4 import EpidemicModel, action_to_u0, action_to_u1
from train_gpu import _config_args
import torch
import matplotlib.pyplot as plt

from utils.normalization import Normalization


def collect_dataset(args, repeat_times=1, env_count=100, agent=None, state_norm=None):
    """Generate dataset from environment"""
    env = EpidemicModel(args, env_count=env_count)
    env.seed()
    device = env.device

    imperfect_obs_dataset = []
    history_action_dataset = []
    true_state_dataset = []

    for _ in range(repeat_times):
        s = env.reset()
        done = False
        while not done:
            # a = torch.zeros((env.env_count, env.ZONE_NUM), device=device)
            # Randomly select action
            if agent is None:
                a = torch.randint(0, env.action_dim, (env.env_count, env.ZONE_NUM), device=device)
            else:
                # Use ppo-agent to select action
                if state_norm is not None:
                    s = state_norm(s, update=False)
                # a, _ = agent.choose_action(s) # choose_action samples according to probability, has randomness
                a = agent.evaluate(s)
            s_, r, done, _ = env.step(a)
            s = s_
        env.render(f"Plot after data collection, repeat times: {repeat_times}")
        # Incomplete observation
        # Calculate incomplete observation info from history_local_obs (current, new)
        # imperfect_obs = env.history_local_obs[:, :, :, :2] * env.POP.unsqueeze(1).unsqueeze(-1) # (env_count, period + 1, ZONE_NUM, 2)
        imperfect_obs = env.extract_observe() # (env_count, period + 1, ZONE_NUM, 2)
        history_action = env.actions # (env_count, period + 1, ZONE_NUM)
        # True (current, new)
        curr_EI = env.simRes[:, :, :, [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(dim=-1)
        # new_EI = env.simRes[:, :, :, env.E_undetected]
        new_EI = env.daily_new_E    # (env_count, period + 1, ZONE_NUM)
        true_state = torch.stack([curr_EI, new_EI], dim=-1) # (env_count, period + 1, ZONE_NUM, 2)

        imperfect_obs_dataset.append(imperfect_obs)
        history_action_dataset.append(history_action)
        true_state_dataset.append(true_state)


    imperfect_obs_dataset = torch.cat(imperfect_obs_dataset, dim=0) # (total_episodes, period + 1, ZONE_NUM)
    history_action_dataset = torch.cat(history_action_dataset, dim=0) # (total_episodes, period + 1, ZONE_NUM)
    true_state_dataset = torch.cat(true_state_dataset, dim=0) # (total_episodes, period + 1, ZONE_NUM)

    # plot_all_env(imperfect_obs_dataset, true_state_dataset)
    #
    # for i in range(20):
    #     plot_one_region(imperfect_obs_dataset, true_state_dataset, region_idx=i)


    dataset = {
        'imperfect_obs': imperfect_obs_dataset,
        'history_action': history_action_dataset,
        'true_state': true_state_dataset,
    }

    print("Dataset generated:\n imperfect_obs:", imperfect_obs_dataset.shape,
          "\n history_action:", history_action_dataset.shape,
          "\n true_state:", true_state_dataset.shape)
    print("In dataset:\n", "Observation: actual values (not divided by POP) \n", "Action: action indices \n", "True state: prediction target, actual values (not divided by POP) ")

    return dataset

def plot_all_env(imperfect_obs_dataset, true_state_dataset):
    # Look at collected data
    # Comparison 1
    first_vector_1 = imperfect_obs_dataset[:, :, :, 0].sum(dim=2).mean(dim=0)  # shape (120,)
    first_vector_2 = true_state_dataset[:, :, :, 0].sum(dim=2).mean(dim=0)

    # Comparison 2
    second_vector_1 = imperfect_obs_dataset[:, :, :, 1].sum(dim=2).mean(dim=0)  # shape (120,)
    second_vector_2 = true_state_dataset[:, :, :, 1].sum(dim=2).mean(dim=0)

    # Plot figures
    plot_4_compare(first_vector_1, first_vector_2, second_vector_1, second_vector_2)

def plot_one_region(imperfect_obs_dataset, true_state_dataset, region_idx=0):
    idx = 0
    # Comparison 1
    first_vector_1 = imperfect_obs_dataset[idx, :, region_idx, 0]  # shape (120,)
    first_vector_2 = true_state_dataset[idx, :, region_idx, 0]

    # Comparison 2
    second_vector_1 = imperfect_obs_dataset[idx, :, region_idx, 0]  # shape (120,)
    second_vector_2 = true_state_dataset[idx, :, region_idx, 0]

    # Plot figures
    plot_4_compare(first_vector_1, first_vector_2, second_vector_1, second_vector_2)




def plot_4_compare(first_vector_1, first_vector_2, second_vector_1, second_vector_2):
    plt.figure(figsize=(10, 8))
    # First comparison (first pair of curves)
    plt.subplot(2, 1, 1)
    plt.plot(first_vector_1.cpu().numpy(), label="obs States - E/I (0)", color='blue')
    plt.plot(first_vector_2.cpu().numpy(), label="SimRes - E/I (Sum)", color='green')
    plt.title("Comparison 1: Rebuild States vs SimRes")
    plt.xlabel("Index (Time Step)")
    plt.ylabel("Value")
    plt.legend()

    # Second comparison (second pair of curves)
    plt.subplot(2, 1, 2)
    plt.plot(second_vector_1.cpu().numpy(), label="obs States - E/I (1)", color='red')
    plt.plot(second_vector_2.cpu().numpy(), label="Daily New Total EI", color='orange')
    plt.title("Comparison 2: Rebuild States vs Daily New Total EI")
    plt.xlabel("Index (Time Step)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _get_history_padding(datas, day, window_size):
    """
    Get padded history window data.

    When day is greater than or equal to window_size, extract the most recent `window_size` days of data.
    When day is less than window_size, pad zeros at the front of the time dimension to ensure the returned window size is `window_size`.

    Args:
        datas (torch.Tensor): Input observation data, shape (env_count, total_days, ZONE_NUM).
        day (int): Current day (timestep), starting from 1.
        window_size (int): Window size, indicating the number of days to look back.

    Returns:
        torch.Tensor: Padded window data, shape (env_count, window_size, ZONE_NUM).

    Note:
        The returned data does not include data at index day
    """
    if day >= window_size:
        # Get the most recent `window_size` days of data
        window = datas[:, (day - window_size):day, :]  # Assume the 3rd feature is the needed observation
    else:
        # Calculate the number of days to pad
        padding = window_size - day
        # Get the available days of data
        current_data = datas[:, :day, :]  # shape: (env_count, day, ZONE_NUM)
        # Pad zeros at the front of the time dimension
        pad = [0] * 2 * len(current_data.shape)
        pad[-4] = padding
        pad = tuple(pad)
        window = torch.nn.functional.pad(current_data, pad, "constant", 0)  # shape: (env_count, window_size, ZONE_NUM)

    return window


def _load_dataset(dataset_dir):
    # Read all files in the directory
    # Get all file paths starting with 'dataset' and ending with '.pth'
    dataset_files = glob.glob(os.path.join(dataset_dir, 'dataset*.pth'))
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files found in directory '{dataset_dir}'.")

    # Initialize data lists
    imperfect_obs_list = []
    history_action_list = []
    true_state_list = []

    # Iterate through all dataset files and load data
    for file in dataset_files:
        data = torch.load(file)
        imperfect_obs_list.append(data['imperfect_obs'])
        history_action_list.append(data['history_action'])
        true_state_list.append(data['true_state'])


    # Merge all data lists
    imperfect_obs = torch.cat(imperfect_obs_list, dim=0)  # shape: (total_samples, period + 1, ZONE_NUM)
    history_action = torch.cat(history_action_list, dim=0)  # shape: (total_samples, period + 1, ZONE_NUM)
    true_state = torch.cat(true_state_list, dim=0)  # shape: (total_samples, period + 1, ZONE_NUM)


    # Build final dataset dictionary
    dataset = {
        'imperfect_obs': imperfect_obs,
        'history_action': history_action,
        'true_state': true_state,
    }


    # Print dataset information
    print("Dataset loaded:")
    print(f"  imperfect_obs: {imperfect_obs.shape}")
    print(f"  history_action: {history_action.shape}")
    print(f"  true_state: {true_state.shape}")

    return dataset



def test(trainer, args, agent, state_norm):
    env = EpidemicModel(args, env_count=10)
    env.seed()
    device = env.device

    rebuild_states = torch.zeros((env.env_count, env.period + 1, env.ZONE_NUM, 2), device=device)

    s = env.reset()
    done = False
    while not done:
        history_action = _get_history_padding(env.actions, env.day - 1, 7)
        obs = _get_history_padding(env.history_local_obs[:, :, :, :2], env.day, 7) * env.POP.unsqueeze(
            1).unsqueeze(-1)
        # Rebuild information
        pre = trainer.predict(obs, history_action)
        pre = torch.clip(pre, min=0)
        rebuild_states[:, env.day - 1, :, :] = pre
        # rebuild_states[:, env.day - 1, :, :] = env.history_local_obs[:, env.day - 1, :, :2] * env.POP.unsqueeze(-1)
        r_s = _get_history_padding(rebuild_states / env.POP.unsqueeze(1).unsqueeze(-1), env.day, env.WINDOW_SIZE)
        r_s = torch.cat((r_s, s[:, :, :, 2:]), dim=-1)
        s = r_s

        if args.use_state_norm:
            s = state_norm(s, update=False)
        a = agent.evaluate(s)
        # a, _ = agent.choose_action(s)  # Sample action
        # a = torch.randint(0, env.action_dim, (env.env_count, env.ZONE_NUM), device=env.device)

        s_, r, done, _ = env.step(a)

        s = s_

    if args.show_fig:
        env.render()
        from uncertainty.obs_imperfect.evaluate_info_rebuild import plot_all_env
        plot_all_env(rebuild_states, env)
        # for i in range(20):
        #     plot_one_region(rebuild_states, env, env_idx=0, region_idx=i)
        # print("Comparison: ", rebuild_states[0, :, :, 0].sum(dim=1), env.simRes[0, :, :, [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(dim=[-2,-1]))
        # print("Comparison: ", rebuild_states[0, :, :, 1].sum(dim=1), env.daily_new_total_EI[0, :, :].sum(dim=-1))
        #
        # # Comparison 1
        # first_vector_1 = rebuild_states[:, :, :, 0].sum(dim=2).mean(dim=0)  # shape (120,)
        # first_vector_2 = env.simRes[:, :, :, [env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(dim=[-2,-1]).mean(dim=0)
        # first_vector_3 = env.simRes[:, :, :, [env.E_detected, env.I_detected, env.I_reported]].sum(dim=[-2,-1]).mean(dim=0)
        #
        # # Comparison 2
        # second_vector_1 = rebuild_states[:, :, :, 1].sum(dim=2).mean(dim=0)  # shape (120,)
        # second_vector_2 = env.daily_new_total_EI[:, :, :].sum(dim=-1).mean(dim=0)  # shape (120,)
        # second_vector_3 = env.daily_estimate_new_EI[:, :, :].sum(dim=-1).mean(dim=0)
        #
        # # Plot figures
        # plt.figure(figsize=(10, 8))
        #
        # # First comparison (first pair of curves)
        # plt.subplot(2, 1, 1)
        # plt.plot(first_vector_1.cpu().numpy(), label="Rebuild States - E/I (0)", color='blue')
        # plt.plot(first_vector_2.cpu().numpy(), label="SimRes - E/I (Sum)", color='green')
        # plt.plot(first_vector_3.cpu().numpy(), label="SimRes - E/I (obs)", color='orange')
        # plt.title("Comparison 1: Rebuild States vs SimRes")
        # plt.xlabel("Index (Time Step)")
        # plt.ylabel("Value")
        # plt.legend()
        #
        # # Second comparison (second pair of curves)
        # plt.subplot(2, 1, 2)
        # plt.plot(second_vector_1.cpu().numpy(), label="Rebuild States - E/I (1)", color='red')
        # plt.plot(second_vector_2.cpu().numpy(), label="Daily New Total EI", color='orange')
        # plt.plot(second_vector_3.cpu().numpy(), label="Daily Estimate New EI (obs)", color='purple')
        # # plt.plot(second_vector_4.cpu().numpy(), label="Daily Detect EI", color='blue')
        # plt.title("Comparison 2: Rebuild States vs Daily New Total EI")
        # plt.xlabel("Index (Time Step)")
        # plt.ylabel("Value")
        # plt.legend()
        #
        # # Show figure
        # plt.tight_layout()
        # plt.show()

    # # TODO: Plot incomplete observation: env.imperfect_obs[:, :, :, 2]  true_state: env.simRes[:, :, :, 2]  rebuild_states: rebuild_states
    # imperfect_obs = env.imperfect_states[:, :, :, 2]
    # true_state = env.simRes[:, :, :, 2]
    # rebuild_states = rebuild_states * env.state_standard_scale # Restore to actual population
    #
    # # Create time axis
    # time_steps = range(env.period + 1)
    #
    # # 1. Plot average across all environments and regions
    # plot_all_env(time_steps, imperfect_obs, true_state, rebuild_states)
    #
    # # 2. Plot average for one environment (across all regions)
    # plot_one_env(time_steps, imperfect_obs, true_state, rebuild_states, env_index=0)
    #
    # # 3. Plot values for one environment and one region
    # for i in range(env.ZONE_NUM):
    #     plot_one_region(time_steps, imperfect_obs, true_state, rebuild_states, env_index=0, zone_index=i)



def main_gru_gnn(args):
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNN, RebuildGruGNNModel
    # args.device_name = 'cpu'

    # Initialize an environment, use its OD POP data, and action_to_u method
    env_temp = EpidemicModel(args, env_count=1)

    # 2. Load agent
    args.use_obs_imperfect = False  # Load agent from deterministic environment
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    print("load directory:", agent.directory)
    agent.load(18)
    dataset_dir = os.path.join(agent.directory, 'dataset_gnn')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # 3. State normalization state_norm loading
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 4. Define dataset collection
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    # # Dataset collection: 1. Data from agent's actions
    # for model_idx in range(1, 101):
    #     agent.load(model_idx)
    #     print(f'model_idx: {model_idx}')
    #     dataset = collect_dataset(args, repeat_times=1, env_count=2, agent=agent, state_norm=state_norm)
    #     # Save to dataset directory under agent.directory, named dataset_{model_idx}
    #     torch.save(dataset, dataset_dir + f'/dataset{model_idx}.pth')
    # dataset = _load_dataset(dataset_dir)
    # agent.load(18)
    # dataset = collect_dataset(args, repeat_times=1, env_count=20, agent=agent, state_norm=state_norm)

    # Dataset collection: 2. Random action dataset
    dataset = collect_dataset(args, repeat_times=1, env_count=20)

    # Get some dimension info of the dataset
    num_samples, total_days, zone_num, _ = dataset['imperfect_obs'].shape
    od_zone_num = env_temp.ZONE_NUM  # Number of zones in OD flow data
    assert zone_num == od_zone_num, "Zone count mismatch"

    # 5. Define prediction model
    # 5.1. Calculate input and output dimensions
    seq_len = 7
    hidden_size = 128
    output_size = 1
    num_layers = 2

    # 5.2. Initialize model
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers, env=env_temp, device_name=args.device_name)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
    # 5.3. Train model
    trainer.train(dataset, num_epochs=100, batch_size=64, patience=5)

    # Save model
    trainer.save('rebuild_gru_gnn_model.pth')

    trainer.load('rebuild_gru_gnn_model.pth')

    # 6. Plot test rebuild results
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1.0
    args.show_fig = True
    test(trainer, args, agent, state_norm=state_norm)


def main_gru_gnn_v1(args):
    from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNN, RebuildGruGNNModel
    # args.device_name = 'cpu'

    # Initialize an environment, use its OD POP data, and action_to_u method
    env_temp = EpidemicModel(args, env_count=1)

    # 2. Load agent
    args.use_obs_imperfect = False  # Load agent from deterministic environment
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    print("load directory:", agent.directory)
    agent.load(args.model_idx)
    dataset_dir = os.path.join(agent.directory, 'dataset_gnn')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # 3. State normalization state_norm loading
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 4. Define dataset collection
    args.use_obs_imperfect = False
    dataset1 = collect_dataset(args, repeat_times=1, env_count=20, agent=agent, state_norm=state_norm)

    # Dataset collection: 2. Random action dataset
    dataset2 = collect_dataset(args, repeat_times=1, env_count=20)

    dataset = dataset1

    # Get some dimension info of the dataset
    num_samples, total_days, zone_num, _ = dataset['imperfect_obs'].shape
    od_zone_num = env_temp.ZONE_NUM  # Number of zones in OD flow data
    assert zone_num == od_zone_num, "Zone count mismatch"

    # Define method, train three types of GNN:
    def do_train(agent, args, dataset, env_temp, hidden_size, num_layers, output_size, seq_len, state_norm, model_name):
        # 5.2. Initialize model
        model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                                   env=env_temp, device_name=args.device_name)
        trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len)
        # 5.3. Train model
        trainer.train(dataset, num_epochs=100, batch_size=64, patience=5)
        # Save model
        model_path = os.path.join(agent.directory, model_name)
        trainer.save(model_path)
        trainer.load(model_path)
        # 6. Plot test rebuild results
        args.use_obs_imperfect = True
        args.obs_imperfect_down = 0.1
        args.obs_imperfect_up = 1.0
        args.show_fig = True
        test(trainer, args, agent, state_norm=state_norm)

    # 5. Define prediction model
    # 5.1. Calculate input and output dimensions
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2

    # # Dataset collected from agent actions
    do_train(agent, args, dataset1, env_temp, hidden_size, num_layers, output_size, seq_len, state_norm, 'rebuild_gru_gnn_model_v1_agent.pth')
    # # Dataset collected from random actions
    do_train(agent, args, dataset2, env_temp, hidden_size, num_layers, output_size, seq_len, state_norm, 'rebuild_gru_gnn_model_v1_random.pth')
    # Ordinary GNN
    env_temp.POP = None
    do_train(agent, args, dataset2, env_temp, hidden_size, num_layers, output_size, seq_len, state_norm, 'rebuild_gru_gnn_model_v1_ordinary.pth')





def main_gru_gnn_with_action(args):
    from uncertainty.obs_imperfect.gru_gnn_model_v2 import RebuildGruGNN, RebuildGruGNNModel
    # args.device_name = 'cpu'

    # Initialize an environment, use its OD POP data, and action_to_u method
    env_temp = EpidemicModel(args, env_count=1)

    # 2. Load agent
    args.max_train_steps = 2400 * 20
    args.use_obs_imperfect = False  # Load agent from deterministic environment
    agent = PPO_discrete_gpu(args)
    agent.directory = '../../' + agent.directory
    print("load directory:", agent.directory)
    agent.load(args.model_idx)
    dataset_dir = os.path.join(agent.directory, 'dataset_gnn')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # 3. State normalization state_norm loading
    args.use_state_norm = True
    if args.use_state_norm:
        state_norm = Normalization(shape=(args.zone_num, args.local_obs_dim * 2), device_name=args.device_name)
        state_norm.load(agent.directory, filename='state_norm.pth')

    # 4. Define dataset collection
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    # Dataset collection: 1. Data from agent's actions
    # for model_idx in range(1, 101):
    #     agent.load(model_idx)
    #     print(f'model_idx: {model_idx}')
    #     dataset = collect_dataset(args, repeat_times=1, env_count=2, agent=agent, state_norm=state_norm)
    #     # Save to dataset directory under agent.directory, named dataset_{model_idx}
    #     torch.save(dataset, dataset_dir + f'/dataset{model_idx}.pth')
    # dataset = _load_dataset(dataset_dir)
    agent.load(args.model_idx)
    args.use_obs_imperfect = False
    dataset1 = collect_dataset(args, repeat_times=1, env_count=20, agent=agent, state_norm=state_norm)

    # Dataset collection: 2. Random action dataset
    dataset2 = collect_dataset(args, repeat_times=1, env_count=20)

    # # Concatenate both datasets
    # dataset = {}
    # for key in dataset1.keys():
    #     dataset[key] = torch.cat([dataset1[key], dataset2[key]], dim=0)

    dataset = dataset1

    # Get some dimension info of the dataset
    num_samples, total_days, zone_num, _ = dataset['imperfect_obs'].shape
    od_zone_num = env_temp.ZONE_NUM  # Number of zones in OD flow data
    assert zone_num == od_zone_num, "Zone count mismatch"

    # 5. Define prediction model
    # 5.1. Calculate input and output dimensions
    seq_len = 7
    hidden_size = 128
    output_size = 2
    num_layers = 2

    # 5.2. Initialize model
    model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                               env=env_temp, device_name=args.device_name, node_output_size=48)
    trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=seq_len, env=env_temp)
    # 5.3. Train model
    trainer.train(dataset, num_epochs=100, batch_size=64, patience=5)

    # Save model
    model_path = os.path.join(agent.directory, 'rebuild_gru_gnn_model_with_action.pth')
    trainer.save(model_path)

    trainer.load(model_path)

    # 6. Plot test rebuild results
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1.0
    args.show_fig = True
    test(trainer, args, agent, state_norm=state_norm)


if __name__ == '__main__':
    from config import args
    _config_args()
    args.env_data_dir = '../../data/'
    args.daily_imported_cases = 0
    args.model_idx = 18

    args.collect_action = True

    # # Low R0 experiment
    # args.ODE_beta = 0.4
    # args.R0 = 'low'

    # main_gru_gnn(args)
    main_gru_gnn_v1(args)
    # main_gru_gnn_with_action(args)