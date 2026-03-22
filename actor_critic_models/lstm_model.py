import torch
import torch.nn as nn
import torch.nn.init as init

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def orthogonal_init_lstm(lstm_layer, gain=1):
    """Orthogonal initialization for LSTM layer weights."""
    # Initialize input to hidden layer weights
    init.orthogonal_(lstm_layer.weight_ih_l0, gain=gain)
    init.orthogonal_(lstm_layer.weight_hh_l0, gain=gain)

    # If LSTM is bidirectional, initialize reverse layer weights
    if lstm_layer.bidirectional:
        init.orthogonal_(lstm_layer.weight_ih_l0_reverse, gain=gain)
        init.orthogonal_(lstm_layer.weight_hh_l0_reverse, gain=gain)

    # Initialize biases
    if lstm_layer.bias:
        init.zeros_(lstm_layer.bias_ih_l0)
        init.zeros_(lstm_layer.bias_hh_l0)
        if lstm_layer.bidirectional:
            init.zeros_(lstm_layer.bias_ih_l0_reverse)
            init.zeros_(lstm_layer.bias_hh_l0_reverse)


class ActorLSTM(nn.Module):
    """
    Actor network using LSTM for sequential data processing.
    
    Input: (B, L, M, H) where B=batch, L=WINDOW_SIZE, M=zone_num, H=local_obs_dim
    Output: (B, M, action_dim) action probability distribution
    """
    def __init__(self, args):
        super(ActorLSTM, self).__init__()
        self.zone_num = args.zone_num
        self.WINDOW_SIZE = args.WINDOW_SIZE
        self.action_dim = args.action_dim

        # Preprocessing layer
        self.pre_fc = nn.Linear(args.local_obs_dim * 2, args.hidden_width)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=args.hidden_width, hidden_size=args.hidden_width, batch_first=True)

        # Hidden and output layers
        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.action_dim)

        # Activation function
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        # Optional: orthogonal initialization
        if args.use_orthogonal_init:
            orthogonal_init(self.pre_fc)
            orthogonal_init_lstm(self.lstm)  # Need custom `orthogonal_init_lstm`
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, s):
        B, L, M, H = s.shape

        # Reshape to (B * M, L, H) for LSTM
        x = s.permute(0, 2, 1, 3).reshape(B * M, L, H)

        # Preprocessing layer
        x = self.activate_func(self.pre_fc(x))

        # LSTM layer
        x, _ = self.lstm(x)

        # Take last timestep output
        x = x[:, -1, :]

        # Hidden and output layers
        x = self.activate_func(self.fc1(x))
        logits = self.fc2(x)

        # Reshape to (B, M, action_dim)
        logits = logits.view(B, M, self.action_dim)

        return logits



class CriticLSTM(nn.Module):
    """
    Critic network for state value estimation using LSTM.
    
    Input: (B, L, M, H) where B=batch, L=WINDOW_SIZE, M=zone_num, H=local_obs_dim
    Output: (B, M) state value
    """
    def __init__(self, args):
        super(CriticLSTM, self).__init__()
        self.zone_num = args.zone_num
        self.WINDOW_SIZE = args.WINDOW_SIZE
        self.action_dim = args.action_dim

        # Preprocessing layer
        self.pre_fc = nn.Linear(args.local_obs_dim * 2, args.hidden_width)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=args.hidden_width, hidden_size=args.hidden_width, batch_first=True)

        # Hidden and output layers
        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, 1)

        # Activation function
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        # Optional: orthogonal initialization
        if args.use_orthogonal_init:
            orthogonal_init(self.pre_fc)
            orthogonal_init_lstm(self.lstm)  # Need custom `orthogonal_init_lstm`
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, s):
        B, L, M, H = s.shape

        # Reshape to (B * M, L, H) for LSTM
        x = s.permute(0, 2, 1, 3).reshape(B * M, L, H)

        # Preprocessing layer
        x = self.activate_func(self.pre_fc(x))

        # LSTM layer
        x, (h_n, c_n) = self.lstm(x)

        # Take last timestep output
        x = x[:, -1, :]

        # Hidden and output layers
        x = self.activate_func(self.fc1(x))
        v_s = self.fc2(x)

        # Reshape to (B, M)
        v_s = v_s.view(B, M)

        return v_s