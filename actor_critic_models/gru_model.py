import torch
import torch.nn as nn
import torch.nn.init as init

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def orthogonal_init_gru(gru_layer, gain=1):
    """Orthogonal initialization for GRU layer weights."""
    # Initialize input to hidden layer weights
    init.orthogonal_(gru_layer.weight_ih_l0, gain=gain)
    init.orthogonal_(gru_layer.weight_hh_l0, gain=gain)

    # If GRU is bidirectional, initialize reverse layer weights
    if gru_layer.bidirectional:
        init.orthogonal_(gru_layer.weight_ih_l0_reverse, gain=gain)
        init.orthogonal_(gru_layer.weight_hh_l0_reverse, gain=gain)

    # Initialize biases
    if gru_layer.bias:
        init.zeros_(gru_layer.bias_ih_l0)
        init.zeros_(gru_layer.bias_hh_l0)
        if gru_layer.bidirectional:
            init.zeros_(gru_layer.bias_ih_l0_reverse)
            init.zeros_(gru_layer.bias_hh_l0_reverse)


class ActorGRU(nn.Module):
    """
    Actor network using GRU for sequential data processing.
    
    Input: (B, L, M, H) where B=batch, L=WINDOW_SIZE, M=zone_num, H=local_obs_dim
    Output: (B, M, action_dim) action probability distribution
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