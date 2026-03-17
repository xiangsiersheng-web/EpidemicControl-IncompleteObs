"""
该类主要暴露给强化学习协同训练时使用
"""
from abc import abstractmethod

import torch
import time
import matplotlib.pyplot as plt
from environment.uncertain_seir_vector_v4 import EpidemicModel

class GenericPredictor:
    def __init__(self, args, model_dir = './'):
        """初始化一个predictor"""
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
    获取填充后的历史窗口数据。

    当天数（day）大于或等于窗口大小（window_size）时，截取最近的 `window_size` 天的数据。
    当天数小于窗口大小时，在时间维度前端填充零，以确保返回的窗口大小为 `window_size`。

    参数:
        datas (torch.Tensor): 输入的观测数据，形状为 (env_count, total_days, ZONE_NUM)。
        day (int): 当前的天数（时间步），从1开始。
        window_size (int): 窗口大小，表示需要回溯的天数。

    返回:
        torch.Tensor: 填充后的窗口数据，形状为 (env_count, window_size, ZONE_NUM)。

    注意：
        返回的数据不包含索引为 day 的数据
    """
    if day >= window_size:
        # 获取最近的 `window_size` 天的数据
        window = datas[:, (day - window_size):day, :]  # 假设第3个特征是需要的观测值
    else:
        # 计算需要填充的天数
        padding = window_size - day
        # 获取已有的天数的数据
        current_data = datas[:, :day, :]  # 形状: (env_count, day, ZONE_NUM)
        # 在时间维度前端填充零
        pad = [0] * 2 * len(current_data.shape)
        pad[-4] = padding
        pad = tuple(pad)
        window = torch.nn.functional.pad(current_data, pad, "constant", 0)  # 形状: (env_count, window_size, ZONE_NUM)

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
        # 1.不完全观测
        imperfect_obs = env.history_local_obs[:, :, :, :2] * env.POP.unsqueeze(1).unsqueeze(-1)  # (env_count, period + 1, ZONE_NUM, 2)
        # 2.动作
        history_action = env.actions  # (env_count, period + 1, ZONE_NUM)
        # 3.真实（现存，新增）
        curr_EI = env.simRes[:, :, :,[env.E_undetected, env.E_detected, env.I_undetected, env.I_detected, env.I_reported]].sum(dim=-1)
        new_EI = env.daily_new_E  # (env_count, period + 1, ZONE_NUM)
        true_state = torch.stack([curr_EI, new_EI], dim=-1)  # (env_count, period + 1, ZONE_NUM, 2)

        dataset = {
            'imperfect_obs': imperfect_obs,
            'history_action': history_action,
            'true_state': true_state,
        }

        print("数据集已生成：\t imperfect_obs:", imperfect_obs.shape,
              "\t history_action:", history_action.shape,
              "\t true_state:", true_state.shape)
        print("数据集中的：\t", "观测：实际数值（未除以POP） \t", "动作：动作下标 \t",
              "真实状态：预测目标，实际数值（未除以POP） ")

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

        # 绘制曲线
        plt.plot(total_EI.cpu().numpy(), label="Total EI(avg)", color='red', linestyle='-', linewidth=2)
        plt.plot(new_EI.cpu().numpy(), label="New EI(avg)", color='orange', linestyle='--', linewidth=2)

        # 添加图例
        plt.legend(loc='upper right', fontsize=10)

        plt.title(title)
        plt.show()



class RebuildGruGNNPredictorV2(BasePredictor):
    """
    使用ODE-GCN-GRU预测，V2包含动作
    """
    def __init__(self, args, model_dir = './'):
        super().__init__(args, model_dir)
        from uncertainty.obs_imperfect.gru_gnn_model_v2 import RebuildGruGNNModel, RebuildGruGNN
        self.seq_len = 7
        hidden_size = 128
        output_size = 2
        num_layers = 2
        env_temp = EpidemicModel(args, env_count=1)

        # 初始化模型
        model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                                   env=env_temp, device_name=args.device_name, node_output_size=32)
        self.trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=self.seq_len, env=env_temp)

        self.rebuild_states = torch.zeros((args.env_count, args.ODE_period + 1, args.zone_num, 2), device=torch.device(args.device_name))

        self.model_path = model_dir + f"/{args.predictor_type}.pth"

    def predict(self, env, s):
        """根据观测和动作调用trainer预测"""
        if env.day == 1:
            torch.fill(self.rebuild_states, 0)
        if not hasattr(self, 'POP'):
            self.POP = env.POP.clone()
        # 获取 obs 和 actions
        history_action = _get_history_padding(env.actions, env.day - 1, self.seq_len)
        obs = _get_history_padding(env.history_local_obs[:, :, :, :2], env.day, self.seq_len) * env.POP.unsqueeze(
            1).unsqueeze(-1)
        # 重建信息
        pre = self.trainer.predict(obs, history_action)
        pre = torch.clip(pre, min=0)
        self.rebuild_states[:, env.day - 1, :, :] = (pre / self.POP.unsqueeze(-1))
        r_s = _get_history_padding(self.rebuild_states, env.day, env.WINDOW_SIZE)
        r_s = torch.cat((r_s, s[:, :, :, 2:]), dim=-1)
        return r_s

    def train(self, env):
        """根据env获取数据集，训练trainer"""
        print("开始训练...")
        # self.trainer.model.reset_top_layer()    # 重置模型
        # self.trainer.model.reset_all_layers()   # 重置模型
        start_time = time.time()
        # 从env中构造数据集
        dataset = self._get_dataset(env)
        self.trainer.train(dataset, num_epochs=50, batch_size=64, patience=5)
        print(f"训练结束，耗时：{time.time() - start_time:.2f} 秒")

    def render(self, title):
        """将重建的状态绘制出来"""
        if not hasattr(self, 'POP'):
            self._plot_rebuild_state(self.rebuild_states, title=title)
        else:
            self._plot_rebuild_state(self.rebuild_states * self.POP.unsqueeze(1).unsqueeze(-1), title=title)

    def save(self, idx):
        model_path = self.model_path.replace('.pth', f"_{idx}.pth")
        print(f"开始保存DL模型: {model_path}")
        self.trainer.save(model_path)

    def load(self, idx):
        model_path = self.model_path.replace('.pth', f"_{idx}.pth")
        print(f"开始加载DL模型: {model_path}")
        self.trainer.load(model_path)

class RebuildGruGNNPredictorV1(BasePredictor):
    """
    使用ODE-GCN-GRU预测，V1不包含动作
    """
    def __init__(self, args, model_dir = './'):
        super().__init__(args, model_dir)
        from uncertainty.obs_imperfect.gru_gnn_model_v1 import RebuildGruGNNModel, RebuildGruGNN
        self.seq_len = 7
        hidden_size = 128
        output_size = 2
        num_layers = 2
        env_temp = EpidemicModel(args, env_count=1)

        # 初始化模型
        model = RebuildGruGNNModel(gru_hidden_size=hidden_size, mlp_output_size=output_size, gru_num_layers=num_layers,
                                   env=env_temp, device_name=args.device_name)
        self.trainer = RebuildGruGNN(model=model, device_name=args.device_name, seq_len=self.seq_len)

        self.rebuild_states = torch.zeros((args.env_count, args.ODE_period + 1, args.zone_num, 2), device=torch.device(args.device_name))

        self.model_path = model_dir + f"/{args.predictor_type}.pth"

    def predict(self, env, s):
        """根据观测和动作调用trainer预测"""
        if env.day == 1:
            torch.fill(self.rebuild_states, 0)
        if not hasattr(self, 'POP'):
            self.POP = env.POP.clone()
        # 获取 obs 和 actions
        history_action = _get_history_padding(env.actions, env.day - 1, self.seq_len)
        obs = _get_history_padding(env.history_local_obs[:, :, :, :2], env.day, self.seq_len) * env.POP.unsqueeze(
            1).unsqueeze(-1)
        # 重建信息
        pre = self.trainer.predict(obs, history_action)
        pre = torch.clip(pre, min=0)
        self.rebuild_states[:, env.day - 1, :, :] = (pre / self.POP.unsqueeze(-1))
        r_s = _get_history_padding(self.rebuild_states, env.day, env.WINDOW_SIZE)
        r_s = torch.cat((r_s, s[:, :, :, 2:]), dim=-1)
        return r_s

    def train(self, env):
        """根据env获取数据集，训练trainer"""
        print("开始训练...")
        start_time = time.time()
        # 从env中构造数据集
        dataset = self._get_dataset(env)
        self.trainer.train(dataset, num_epochs=50, batch_size=64, patience=5)
        print(f"训练结束，耗时：{time.time() - start_time:.2f} 秒")

    def render(self, title):
        """将重建的状态绘制出来"""
        if not hasattr(self, 'POP'):
            self._plot_rebuild_state(self.rebuild_states, title=title)
        else:
            self._plot_rebuild_state(self.rebuild_states * self.POP.unsqueeze(1).unsqueeze(-1), title=title)

    def save(self, idx):
        model_path = self.model_path.replace('.pth', f"_{idx}.pth")
        print(f"开始保存DL模型: {model_path}")
        self.trainer.save(model_path)

    def load(self, idx):
        model_path = self.model_path.replace('.pth', f"_{idx}.pth")
        print(f"开始加载DL模型: {model_path}")
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



