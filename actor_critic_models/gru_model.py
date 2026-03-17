import torch
import torch.nn as nn
import torch.nn.init as init

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def orthogonal_init_gru(gru_layer, gain=1):
    """
    正交初始化 GRU 层的权重。
    """
    # 初始化输入到隐藏层权重
    init.orthogonal_(gru_layer.weight_ih_l0, gain=gain)
    init.orthogonal_(gru_layer.weight_hh_l0, gain=gain)

    # 如果 GRU 是双向的，初始化反向层的权重
    if gru_layer.bidirectional:
        init.orthogonal_(gru_layer.weight_ih_l0_reverse, gain=gain)
        init.orthogonal_(gru_layer.weight_hh_l0_reverse, gain=gain)

    # 初始化偏置
    if gru_layer.bias:
        init.zeros_(gru_layer.bias_ih_l0)
        init.zeros_(gru_layer.bias_hh_l0)
        if gru_layer.bidirectional:
            init.zeros_(gru_layer.bias_ih_l0_reverse)
            init.zeros_(gru_layer.bias_hh_l0_reverse)


class ActorGRU(nn.Module):
    """
    Actor 网络，基于 GRU 处理时间序列数据并输出动作概率分布。

    网络结构：
    1. 输入层：状态输入，形状为 (B, L, M, H)
        - B: 批量大小 (batch size)
        - L: 时间序列长度 (WINDOW_SIZE)
        - M: 区域数量 (zone_num)
        - H: 每个区域的特征维度 (local_obs_dim)
    2. 预处理层 (pre_fc)：将输入特征映射到 GRU 的输入维度。
    3. GRU 层：提取时间序列的动态特征。
    4. 全连接隐藏层 (fc1)：将 GRU 输出进一步映射到高维空间。
    5. 输出层 (fc2)：生成每个区域的动作概率分布，输出形状为 (B, M, action_dim)。

    前向传播：
    - 接收输入状态张量 s，经过预处理、GRU 和全连接层，输出各区域动作概率分布。
    """
    def __init__(self, args):
        super(ActorGRU, self).__init__()
        self.zone_num = args.zone_num
        self.WINDOW_SIZE = args.WINDOW_SIZE
        self.action_dim = args.action_dim
        # self.action_type = args.action_type
        self.local_obs_dim = args.local_obs_dim

        self.gru = nn.GRU(input_size=args.local_obs_dim, hidden_size=args.hidden_width, batch_first=True)
        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.action_dim)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            # print("------use_orthogonal_init------")
            orthogonal_init_gru(self.gru)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, s):
        s = s[:, :, :, :self.local_obs_dim]
        B, L, M, H = s.shape
        x = s.permute(0, 2, 1, 3).reshape(B * M, L, H)

        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.activate_func(self.fc1(x))
        logits = self.fc2(x)
        logits = logits.view(B, M, self.action_dim)

        return logits


class CriticGRU(nn.Module):

    def __init__(self, args):
        super(CriticGRU, self).__init__()
        self.zone_num = args.zone_num
        self.WINDOW_SIZE = args.WINDOW_SIZE
        self.action_dim = args.action_dim

        self.gru = nn.GRU(input_size=args.local_obs_dim * 2, hidden_size=args.hidden_width, batch_first=True)
        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, 1)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            # print("------use_orthogonal_init------")
            orthogonal_init_gru(self.gru)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)


    def forward(self, s):
        B, L, M, H = s.shape
        x = s.permute(0, 2, 1, 3).reshape(B * M, L, H)

        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.activate_func(self.fc1(x))
        v_s = self.fc2(x)
        v_s = v_s.view(B, M)
        return v_s