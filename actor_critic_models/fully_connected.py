import torch
import torch.nn as nn

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    """Actor network for action probability distribution."""
    def __init__(self, args):
        super(Actor, self).__init__()
        self.zone_num = args.zone_num
        self.action_dim = args.action_dim
        self.local_obs_dim = args.local_obs_dim

        # Create network layers
        self.action_net = self._create_network(args)

        if args.use_orthogonal_init:
            # Orthogonal initialization
            orthogonal_init(self.action_net.fc1)
            orthogonal_init(self.action_net.fc2)
            orthogonal_init(self.action_net.fc3)

    def _create_network(self, args):
        """Create network structure."""
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
    """Critic network for state value estimation."""
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
