import torch


class ReplayBufferTensor:
    def __init__(self, args):
        # Initialize tensors for storing data
        self.device = torch.device(args.device_name)
        self.s = torch.zeros((args.batch_size, args.WINDOW_SIZE, args.zone_num, args.local_obs_dim * 2)).to(self.device)
        self.s_ = torch.zeros_like(self.s)
        self.a = torch.zeros((args.batch_size, args.zone_num)).to(self.device)
        self.a_logprob = torch.zeros((args.batch_size, args.zone_num)).to(self.device)
        self.r = torch.zeros((args.batch_size, args.zone_num)).to(self.device)
        # 可以确定所有区域会同时结束，所以不必增加zone_num维度
        self.dw = torch.zeros((args.batch_size, 1)).to(self.device)
        self.done = torch.zeros((args.batch_size, 1)).to(self.device)
        self.count = 0  # 记录当前存储的数据数量
        self.capacity = args.batch_size  # 存储最大容量
        self.env_count = int(args.env_count)  # 并行环境数量
        self.period = args.ODE_period  # 环境运行的天数

    def store(self, idx, s, a, a_logprob, r, s_, dw, done):
        """ 根据索引存储数据"""
        # 指定索引位置存储数据
        if idx < self.capacity:  # 确保索引在容量范围内
            self.s[idx] = s
            self.a[idx] = a
            self.a_logprob[idx] = a_logprob
            self.r[idx] = r
            self.s_[idx] = s_
            self.dw[idx] = dw
            self.done[idx] = done
            self.count += 1

    def store_batch(self, idxs, s, a, a_logprob, r, s_, dw, done):
        """批量存储数据, 但dw和done是标量"""
        for i in range(idxs.shape[0]):
            idx = idxs[i]
            if idx < self.capacity:  # 确保索引在容量范围内
                self.s[idx] = s[i]
                self.a[idx] = a[i]
                self.a_logprob[idx] = a_logprob[i]
                self.r[idx] = r[i]
                self.s_[idx] = s_[i]
                self.dw[idx] = dw
                self.done[idx] = done
                self.count += 1

    def get_stored_data(self):
        # 返回所有存储的数据，不打乱顺序
        return self.s, self.a, self.a_logprob, self.r, self.s_, self.dw, self.done
