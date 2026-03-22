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
        self.dw = torch.zeros((args.batch_size, 1)).to(self.device)
        self.done = torch.zeros((args.batch_size, 1)).to(self.device)
        self.count = 0  # Current stored count
        self.capacity = args.batch_size  # Max capacity
        self.env_count = int(args.env_count)  # Number of parallel environments
        self.period = args.ODE_period  # Number of days the environment runs

    def store(self, idx, s, a, a_logprob, r, s_, dw, done):
        """Store data at given index."""
        if idx < self.capacity:
            self.s[idx] = s
            self.a[idx] = a
            self.a_logprob[idx] = a_logprob
            self.r[idx] = r
            self.s_[idx] = s_
            self.dw[idx] = dw
            self.done[idx] = done
            self.count += 1

    def store_batch(self, idxs, s, a, a_logprob, r, s_, dw, done):
        """Batch store data (dw and done are scalars)."""
        for i in range(idxs.shape[0]):
            idx = idxs[i]
            if idx < self.capacity:
                self.s[idx] = s[i]
                self.a[idx] = a[i]
                self.a_logprob[idx] = a_logprob[i]
                self.r[idx] = r[i]
                self.s_[idx] = s_[i]
                self.dw[idx] = dw
                self.done[idx] = done
                self.count += 1

    def get_stored_data(self):
        """Return all stored data."""
        return self.s, self.a, self.a_logprob, self.r, self.s_, self.dw, self.done
