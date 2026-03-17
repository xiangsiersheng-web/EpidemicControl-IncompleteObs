import torch
import torch.nn as nn
import torch.nn.init as init

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def orthogonal_init_lstm(lstm_layer, gain=1):
    """
    正交初始化 LSTM 层的权重。
    """
    # 初始化输入到隐藏层权重
    init.orthogonal_(lstm_layer.weight_ih_l0, gain=gain)
    init.orthogonal_(lstm_layer.weight_hh_l0, gain=gain)

    # 如果 LSTM 是双向的，初始化反向层的权重
    if lstm_layer.bidirectional:
        init.orthogonal_(lstm_layer.weight_ih_l0_reverse, gain=gain)
        init.orthogonal_(lstm_layer.weight_hh_l0_reverse, gain=gain)

    # 初始化偏置
    if lstm_layer.bias:
        init.zeros_(lstm_layer.bias_ih_l0)
        init.zeros_(lstm_layer.bias_hh_l0)
        if lstm_layer.bidirectional:
            init.zeros_(lstm_layer.bias_ih_l0_reverse)
            init.zeros_(lstm_layer.bias_hh_l0_reverse)


class ActorLSTM(nn.Module):
    """
    Actor 网络，基于 LSTM 处理时间序列数据并输出动作概率分布。

    网络结构：
    1. 输入层：状态输入，形状为 (B, L, M, H)
        - B: 批量大小 (batch size)
        - L: 时间序列长度 (WINDOW_SIZE)
        - M: 区域数量 (zone_num)
        - H: 每个区域的特征维度 (local_obs_dim)
    2. 预处理层 (pre_fc)：将输入特征映射到 LSTM 的输入维度。
    3. LSTM 层：提取时间序列的动态特征。
    4. 全连接隐藏层 (fc1)：将 LSTM 输出进一步映射到高维空间。
    5. 输出层 (fc2)：生成每个区域的动作概率分布，输出形状为 (B, M, action_dim)。

    前向传播：
    - 接收输入状态张量 s，经过预处理、LSTM 和全连接层，输出各区域动作概率分布。
    """
    def __init__(self, args):
        super(ActorLSTM, self).__init__()
        self.zone_num = args.zone_num
        self.WINDOW_SIZE = args.WINDOW_SIZE
        self.action_dim = args.action_dim

        # 预处理层：映射输入到 LSTM 的输入维度
        self.pre_fc = nn.Linear(args.local_obs_dim * 2, args.hidden_width)

        # LSTM 层
        self.lstm = nn.LSTM(input_size=args.hidden_width, hidden_size=args.hidden_width, batch_first=True)

        # 全连接隐藏层和输出层
        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.action_dim)

        # 激活函数
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        # 可选：使用正交初始化
        if args.use_orthogonal_init:
            orthogonal_init(self.pre_fc)
            orthogonal_init_lstm(self.lstm)  # 这里需要自定义 `orthogonal_init_lstm`
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, s):
        B, L, M, H = s.shape

        # 调整张量形状为 (B * M, L, H)，以适应 LSTM 和全连接层
        x = s.permute(0, 2, 1, 3).reshape(B * M, L, H)

        # 预处理层
        x = self.activate_func(self.pre_fc(x))

        # LSTM 层
        x, _ = self.lstm(x)

        # 取 LSTM 的最后一个时间步输出
        x = x[:, -1, :]

        # 全连接隐藏层和输出层
        x = self.activate_func(self.fc1(x))
        logits = self.fc2(x)

        # 调整形状为 (B, M, action_dim)
        logits = logits.view(B, M, self.action_dim)

        return logits



class CriticLSTM(nn.Module):
    """
    Critic 网络，用于估计状态的价值函数 (V)。

    网络结构：
    1. 输入层：输入状态张量 s，形状为 (B, L, M, H)
        - B: 批量大小 (batch size)
        - L: 时间序列长度 (WINDOW_SIZE)
        - M: 区域数量 (zone_num)
        - H: 每个区域的特征维度 (local_obs_dim)
    2. 预处理层 (pre_fc)：将输入特征映射到 LSTM 的输入维度。
    3. LSTM 层：提取时间序列的动态特征。
    4. 全连接隐藏层 (fc1)：进一步处理 LSTM 输出。
    5. 输出层 (fc2)：生成状态价值估计，输出形状为 (B, M, 1)。

    前向传播：
    - 接收输入状态张量 s，经过预处理、LSTM 和全连接层，输出状态价值估计。
    """
    def __init__(self, args):
        super(CriticLSTM, self).__init__()
        self.zone_num = args.zone_num
        self.WINDOW_SIZE = args.WINDOW_SIZE
        self.action_dim = args.action_dim

        # 预处理层
        self.pre_fc = nn.Linear(args.local_obs_dim * 2, args.hidden_width)

        # LSTM 层
        self.lstm = nn.LSTM(input_size=args.hidden_width, hidden_size=args.hidden_width, batch_first=True)

        # 全连接隐藏层和输出层
        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, 1)

        # 激活函数
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        # 可选：使用正交初始化
        if args.use_orthogonal_init:
            orthogonal_init(self.pre_fc)
            orthogonal_init_lstm(self.lstm)  # 这里需要自定义 `orthogonal_init_lstm`
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, s):
        B, L, M, H = s.shape

        # 调整张量形状为 (B * M, L, H)，以适应 LSTM 和全连接层
        x = s.permute(0, 2, 1, 3).reshape(B * M, L, H)

        # 预处理层
        x = self.activate_func(self.pre_fc(x))

        # LSTM 层
        x, (h_n, c_n) = self.lstm(x)

        # 取 LSTM 的最后一个时间步输出
        x = x[:, -1, :]

        # 全连接隐藏层和输出层
        x = self.activate_func(self.fc1(x))
        v_s = self.fc2(x)

        # 调整形状为 (B, M)
        v_s = v_s.view(B, M)

        return v_s