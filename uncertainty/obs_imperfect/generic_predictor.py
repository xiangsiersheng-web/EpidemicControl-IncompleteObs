"""
This class is mainly exposed for use in reinforcement learning collaborative training
"""
from abc import abstractmethod

import torch
import time
import matplotlib.pyplot as plt
from environment.uncertain_seir_vector_v4 import EpidemicModel

class GenericPredictor:
    def __init__(self, args, model_dir = './'):
        """Initialize a predictor"""
        if args.predictor_type == 'rebuild_gru_gnn_model_with_action':
            self.predictor = RebuildGruGNNPredictorV2(args, model_dir)
        elif args.predictor_type == 'rebuild_gru_gnn_model_no_action':
            self.predictor = RebuildGruGNNPredictorV1(args, model_dir)
        elif args.predictor_type == 'none':
            self.predictor = NoPredictionPredictor(args, model_dir)
        else:
            raise ValueError("predictor_type is not valid")

    def predict(self, env, s):
        return self.predictor.predict(env, s)

    def train(self, env):
        self.predictor.train(env)

    def save(self, idx):
        self.predictor.save(idx)

    def load(self, idx):
        self.predictor.load(idx)

    def render(self, title = ''):
        self.predictor.render(title=title)

    def get_trainer(self):
        return self.predictor.trainer

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


class BasePredictor:
    def __init__(self, args, model_dir = './'):
        self.model_path = model_dir + "/none.pth"
        self.device = torch.device(args.device_name)
        self.trainer = None

    @abstractmethod
    def predict(self, env, s):
        pass

    @abstractmethod
    def train(self, env):
        pass

    def _get_dataset(self, env):
        # 1. Incomplete observation
        imperfect_obs = env.history_local_obs[:, :, :, :2] * env.POP.unsqueeze(1).unsqueeze(-1)  # (env_count, period + 1, ZONE_NUM, 2)
        # 2. Actions
        history_action = env.actions  # (env_count, period + 1, ZONE_NUM)
        # 3. True (current, new)
        curr_EI = env.simRes[:, :, :,[env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(dim=-1)
        new_EI = env.daily_new_E  # (env_count, period + 1, ZONE_NUM)
        true_state = torch.stack([curr_EI, new_EI], dim=-1)  # (env_count, period + 1, ZONE_NUM, 2)

        dataset = {
            'imperfect_obs': imperfect_obs,
            'history_action': history_action,
            'true_state': true_state,
        }

        print("Dataset generated:\t imperfect_obs:", imperfect_obs.shape,
              "\t history_action:", history_action.shape,
              "\t true_state:", true_state.shape)
        print("In dataset:\t", "Observation: actual values (not divided by POP) \t", "Action: action indices \t",
              "True state: prediction target, actual values (not divided by POP) ")

        return dataset

    def save(self, idx):
        pass

    def load(self, idx):
        pass

    def render(self, title):
        pass

    def _plot_rebuild_state(self, rebuild_states, title = ''):
        plt.figure(dpi=120, figsize=(7, 5))
        plt.grid(linestyle='-.', axis='both')

        total_EI = rebuild_states[:, :, :, 0].sum(dim=2).mean(dim=0)
        new_EI = rebuild_states[:, :, :, 1].sum(dim=2).mean(dim=0)

        # Plot curves
        plt.plot(total_EI.cpu().numpy(), label="Total EI(avg)", color='red', linestyle='-', linewidth=2)
        plt.plot(new_EI.cpu().numpy(), label="New EI(avg)", color='orange', linestyle='--', linewidth=2)

        # Add legend
        plt.legend(loc='upper right', fontsize=10)

        plt.title(title)
        plt.show()



class RebuildGruGNNPredictorV2(BasePredictor):
    """
    Use ODE-GCN-GRU for prediction, V2 includes actions
    """
    def __init__(self, args, model_dir = './'):
        super().__init__(args, model_dir)
        from uncertainty.obs_imperfect.gru_gnn_model_v2 import RebuildGruGNNModel, RebuildGruGNN
        self.seq_len = 7
        hidden_size = 128
        output_size = 2
        num_layers = 2
        env_temp = EpidemicModel(args, env_count=1)

        # Initialize model
        model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                                   env=env_temp, device_name=args.device_name, node_output_size=32)
        self.trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=self.seq_len, env=env_temp)

        self.rebuild_states = torch.zeros((args.env_count, args.ODE_period + 1, args.zone_num, 2), device=torch.device(args.device_name))

        self.model_path = model_dir + f"/{args.predictor_type}.pth"

    def predict(self, env, s):
        """Call trainer to predict based on observations and actions"""
        if env.day == 1:
            torch.fill(self.rebuild_states, 0)
        if not hasattr(self, 'POP'):
            self.POP = env.POP.clone()
        # Get obs and actions
        history_action = _get_history_padding(env.actions, env.day - 1, self.seq_len)
        obs = _get_history_padding(env.history_local_obs[:, :, :, :2], env.day, self.seq_len) * env.POP.unsqueeze(
            1).unsqueeze(-1)
        # Rebuild information
        pre = self.trainer.predict(obs, history_action)
        pre = torch.clip(pre, min=0)
        self.rebuild_states[:, env.day - 1, :, :] = (pre / self.POP.unsqueeze(-1))
        r_s = _get_history_padding(self.rebuild_states, env.day, env.WINDOW_SIZE)
        r_s = torch.cat((r_s, s[:, :, :, 2:]), dim=-1)
        return r_s

    def train(self, env):
        """Get dataset from env and train trainer"""
        print("Start training...")
        # self.trainer.model.reset_top_layer()    # Reset model
        # self.trainer.model.reset_all_layers()   # Reset model
        start_time = time.time()
        # Construct dataset from env
        dataset = self._get_dataset(env)
        self.trainer.train(dataset, num_epochs=50, batch_size=64, patience=5)
        print(f"Training completed, time elapsed: {time.time() - start_time:.2f} seconds")

    def render(self, title):
        """Plot the rebuilt states"""
        if not hasattr(self, 'POP'):
            self._plot_rebuild_state(self.rebuild_states, title=title)
        else:
            self._plot_rebuild_state(self.rebuild_states * self.POP.unsqueeze(1).unsqueeze(-1), title=title)

    def save(self, idx):
        model_path = self.model_path.replace('.pth', f"_{idx}.pth")
        print(f"Start saving DL model: {model_path}")
        self.trainer.save(model_path)

    def load(self, idx):
        model_path = self.model_path.replace('.pth', f"_{idx}.pth")
        print(f"Start loading DL model: {model_path}")
        self.trainer.load(model_path)

class RebuildGruGNNPredictorV1(BasePredictor):
    """
    Use ODE-GCN-GRU for prediction, V1 does not include actions
    """
    def __init__(self, args, model_dir = './'):
        super().__init__(args, model_dir)
        from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
        self.seq_len = 7
        hidden_size = 128
        output_size = 2
        num_layers = 2
        env_temp = EpidemicModel(args, env_count=1)

        # Initialize model
        model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                                   env=env_temp, device_name=args.device_name)
        self.trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=self.seq_len)

        self.rebuild_states = torch.zeros((args.env_count, args.ODE_period + 1, args.zone_num, 2), device=torch.device(args.device_name))

        self.model_path = model_dir + f"/{args.predictor_type}.pth"

    def predict(self, env, s):
        """Call trainer to predict based on observations and actions"""
        if env.day == 1:
            torch.fill(self.rebuild_states, 0)
        if not hasattr(self, 'POP'):
            self.POP = env.POP.clone()
        # Get obs and actions
        history_action = _get_history_padding(env.actions, env.day - 1, self.seq_len)
        obs = _get_history_padding(env.history_local_obs[:, :, :, :2], env.day, self.seq_len) * env.POP.unsqueeze(
            1).unsqueeze(-1)
        # Rebuild information
        pre = self.trainer.predict(obs, history_action)
        pre = torch.clip(pre, min=0)
        self.rebuild_states[:, env.day - 1, :, :] = (pre / self.POP.unsqueeze(-1))
        r_s = _get_history_padding(self.rebuild_states, env.day, env.WINDOW_SIZE)
        r_s = torch.cat((r_s, s[:, :, :, 2:]), dim=-1)
        return r_s

    def train(self, env):
        """Get dataset from env and train trainer"""
        print("Start training...")
        start_time = time.time()
        # Construct dataset from env
        dataset = self._get_dataset(env)
        self.trainer.train(dataset, num_epochs=50, batch_size=64, patience=5)
        print(f"Training completed, time elapsed: {time.time() - start_time:.2f} seconds")

    def render(self, title):
        """Plot the rebuilt states"""
        if not hasattr(self, 'POP'):
            self._plot_rebuild_state(self.rebuild_states, title=title)
        else:
            self._plot_rebuild_state(self.rebuild_states * self.POP.unsqueeze(1).unsqueeze(-1), title=title)

    def save(self, idx):
        model_path = self.model_path.replace('.pth', f"_{idx}.pth")
        print(f"Start saving DL model: {model_path}")
        self.trainer.save(model_path)

    def load(self, idx):
        model_path = self.model_path.replace('.pth', f"_{idx}.pth")
        print(f"Start loading DL model: {model_path}")
        self.trainer.load(model_path)

class NoPredictionPredictor(BasePredictor):
    def __init__(self, args, model_dir = './'):
        super().__init__(args, model_dir)

    def predict(self, env, s):
        return s

    def train(self, env):
        print("NoPredictionPredictor does not need to train.")

    def save(self, idx):
        print("NoPredictionPredictor does not need to save.")

    def load(self, idx):
        print("NoPredictionPredictor does not need to load.")

    def render(self, title):
        print("NoPredictionPredictor does not need to render.")



