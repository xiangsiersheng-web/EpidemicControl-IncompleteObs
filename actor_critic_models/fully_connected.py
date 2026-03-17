import torch
import torch.nn as nn

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    """
    初始化actor网络结构。
    - 输入层：输入状态向量，形状为(B, args.WINDOW_SIZE, zone_NUM, local_state_dim * 2)


    前向传播：通过网络接收输入状态向量s，输出各区域动作概率分布。
    """
    def __init__(self, args):
        super(Actor, self).__init__()
        self.zone_num = args.zone_num
        self.action_dim = args.action_dim
        self.local_obs_dim = args.local_obs_dim

        # 创建三层网络
        self.action_net = self._create_network(args)

        if args.use_orthogonal_init:
            # 对网络的权重进行正交初始化
            orthogonal_init(self.action_net.fc1)
            orthogonal_init(self.action_net.fc2)
            orthogonal_init(self.action_net.fc3)

    def _create_network(self, args):
        """
        创建网络结构。
        """
        return ActionNetwork(args)


    def forward(self, s):
        s = s[:, :, :, :self.local_obs_dim]
        B, L, M, H = s.shape
        s = s.permute(0, 2, 1, 3).reshape(B*M, L*H)

        logits = self.action_net(s)
        logits = logits.reshape(B, M, self.action_dim)

        return logits

class ActionNetwork(nn.Module):
    """

    """
    def __init__(self, args):
        super(ActionNetwork, self).__init__()
        self.local_obs_dim = args.local_obs_dim

        self.fc1 = nn.Linear(args.WINDOW_SIZE * args.local_obs_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh


    def forward(self, x):
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        logits = self.fc3(x)

        return logits


class Critic(nn.Module):
    """
    初始化critic网络结构。
    - 输入层：接收状态向量，形状为(B, window_sim, zone_num, local_state_dim * 2)

    前向传播：得到(B, zone_num)
    """
    def __init__(self, args):
        super(Critic, self).__init__()
        self.zone_num = args.zone_num
        self.action_dim = args.action_dim

        self.fc1 = nn.Linear(args.WINDOW_SIZE * args.local_obs_dim * 2, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            # print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        B, L, M, H = s.shape
        x = s.permute(0, 2, 1, 3).reshape(B * M, L*H)
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        v_s = self.fc3(x)
        v_s = v_s.view(B, M)
        return v_s


# class Actor(nn.Module):
#     """
#     初始化actor网络结构。
#     - 输入层：输入状态向量，形状为(args.WINDOW_SIZE * args.state_dim, 1)
#     - 隐藏层fc1：hidden_width个神经元，输出形状为(hidden_width, 1)
#     - 隐藏层fc2：hidden_width个神经元，输出形状为(hidden_width, 1)
#     - 多头输出层heads：每个头输出动作概率分布，输出形状为(action_dim, 1)
#
#     前向传播：通过网络接收输入状态向量s，输出各区域动作概率分布。
#     """
#
#     def __init__(self, args):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(args.WINDOW_SIZE * args.state_dim, args.hidden_width)
#         self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
#         self.zone_num = args.zone_num
#         self.WINDOW_SIZE = args.WINDOW_SIZE
#         self.action_seq_len = args.action_seq_len
#
#         # 每个头要不要看自己前几天做的决策
#         self.input_prev_action = False if "input_prev_action" not in vars(args) else args.input_prev_action
#
#         if not self.input_prev_action:
#             self.heads = torch.nn.ModuleList([
#                 torch.nn.Linear(args.hidden_width, args.action_dim) for _ in range(self.zone_num)
#             ])
#         else:
#             self.heads = torch.nn.ModuleList([
#                 torch.nn.Linear(args.hidden_width + self.action_seq_len, args.action_dim) for _ in range(self.zone_num)
#             ])
#
#         # self.fc3 = nn.Linear(args.hidden_width, args.action_dim * args.zone_num)
#         self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
#
#         if args.use_orthogonal_init:
#             # print("------use_orthogonal_init------")
#             orthogonal_init(self.fc1)
#             orthogonal_init(self.fc2)
#             for head in self.heads:
#                 orthogonal_init(head, gain=0.01)
#
#     def forward(self, s):
#         I_s, a_s = s[:, :self.zone_num * self.WINDOW_SIZE], s[:, self.zone_num * self.WINDOW_SIZE:]
#         I_s = self.activate_func(self.fc1(I_s))
#         I_s = self.activate_func(self.fc2(I_s))
#         if not self.input_prev_action:
#             logits = [self.heads[i](I_s) for i in range(self.zone_num)]
#         else:
#             a_s = a_s.view(a_s.size(0), self.action_seq_len, self.zone_num)
#             logits = [self.heads[i](torch.cat([I_s, a_s[:, :, i]], dim=1)) for i in range(self.zone_num)]
#
#         # logits = self.fc3(s)
#
#         return logits
#
#
# class Critic(nn.Module):
#     """
#     初始化critic网络结构。
#     - 输入层：接收状态向量，形状为(args.WINDOW_SIZE * args.state_dim, 1)
#     - 隐藏层fc1：包含hidden_width个神经元，输出形状为(hidden_width, 1)
#     - 隐藏层fc2：包含hidden_width个神经元，输出形状为(hidden_width, 1)
#     - 输出层fc3：输出单个值，代表状态的价值估计
#
#     前向传播：该函数接收状态s，通过网络计算并返回状态的价值。
#     """
#
#     def __init__(self, args):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear((args.WINDOW_SIZE + args.action_seq_len) * args.state_dim, args.hidden_width)
#         self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
#         self.fc3 = nn.Linear(args.hidden_width, 1)
#         self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
#
#         if args.use_orthogonal_init:
#             # print("------use_orthogonal_init------")
#             orthogonal_init(self.fc1)
#             orthogonal_init(self.fc2)
#             orthogonal_init(self.fc3)
#
#     def forward(self, s):
#         s = self.activate_func(self.fc1(s))
#         s = self.activate_func(self.fc2(s))
#         v_s = self.fc3(s)
#         return v_s