import numpy as np
import torch


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device_name='cuda:0'):  # shape:the dimension of input data
        self.device = torch.device(device_name)
        self.n = 0
        self.mean = torch.zeros(shape, device=self.device)
        self.S = torch.zeros(shape, device=self.device)
        self.std = torch.ones(shape, device=self.device)

    def update(self, x):
        x = x.float()
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / max(self.n - 1, 1))


class Normalization:
    def __init__(self, shape, device_name='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device_name=device_name)
        self.device_name = device_name

    def __call__(self, x, update=True):
        original_shape = x.shape
        x = x.view(-1, original_shape[-2], original_shape[-1]) # 融合 B, L
        # Ensure input tensor has at least 3 dimensions
        if len(original_shape) < 3:
            raise ValueError(f"Expected input with at least 3 dimensions, got {len(original_shape)}.")

        if update:
            self.running_ms.update(x.mean(dim=0))
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x.view(original_shape)

    def get_original(self, x):
        original_shape = x.shape
        x = x.view(-1, original_shape[-2], original_shape[-1])  # 融合 B, L
        x = (x * self.running_ms.std) + self.running_ms.mean
        return x.view(original_shape)

    def save(self, dir, filename='state_norm.pth'):
        torch.save({'mean': self.running_ms.mean,
                    'std': self.running_ms.std},
                   dir + '/' + filename)
        print('Saved normalization parameters to', dir + '/' + filename)

    def load(self, dir, filename='state_norm.pth'):
        """加载归一化参数"""

        # 加载模型文件时，检查是否有 CUDA 可用
        filepath = dir + '/' + filename
        if torch.cuda.is_available() and "cuda" in self.device_name:  # 如果有 GPU
            state_dict = torch.load(filepath)
        else:  # 如果没有 GPU，强制加载到 CPU
            state_dict = torch.load(filepath, map_location=torch.device('cpu'))

        # 更新归一化参数
        self.running_ms.mean = state_dict['mean']
        self.running_ms.std = state_dict['std']

        print('Loaded normalization parameters from', filepath)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
