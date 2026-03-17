import torch
import numpy as np


class ReplayBufferAsymmetric:
    """
    非对称 actor-critic 架构的 replay buffer
    """
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.WINDOW_SIZE * args.state_dim))
        self.s_noise = np.zeros((args.batch_size, args.WINDOW_SIZE * args.state_dim))
        self.s_ = np.zeros((args.batch_size, args.WINDOW_SIZE * args.state_dim))
        self.s_noise_ = np.zeros((args.batch_size, args.WINDOW_SIZE * args.state_dim))  # 无噪声下一状态
        self.a = np.zeros((args.batch_size, args.zone_num))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, s_noise, a, a_logprob, r, s_, s_noise_, dw, done):
        self.s[self.count] = s
        self.s_noise[self.count] = s_noise
        self.s_[self.count] = s_
        self.s_noise_[self.count] = s_noise_  # 存储无噪声下一状态
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        s_noise = torch.tensor(self.s_noise, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        s_noise_ = torch.tensor(self.s_noise_, dtype=torch.float)  # 转换无噪声下一状态为张量
        a = torch.tensor(self.a, dtype=torch.long)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, s_noise, a, a_logprob, r, s_, s_noise_, dw, done
