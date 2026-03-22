# -*- ecoding: utf-8 -*-
# @ModuleName: ppo_discrete_gpu
# @Function: 
#  
# @Time: 2024/9/30 14:20
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import os

from actor_critic_models.fully_connected import Actor, Critic
from actor_critic_models.lstm_model import ActorLSTM, CriticLSTM
from actor_critic_models.gru_model import ActorGRU, CriticGRU
torch.autograd.set_detect_anomaly(True)

class PPO_discrete_gpu:
    def __init__(self, args):
        """Initialize actor-critic networks, optimizers, and learning rates."""
        args_dict = vars(args)
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.entropy_coef_copy = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.zone_num = args.zone_num

        # Build model save path
        self.directory = self._build_directory(args)

        if "cuda" in args.device_name:
            self.device = torch.device(args.device_name if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        if args.actor_critic_model == "mlp":
            self.actor = Actor(args)
            self.critic = Critic(args)
        elif args.actor_critic_model == "lstm":
            self.actor = ActorLSTM(args)
            self.critic = CriticLSTM(args)
        elif args.actor_critic_model == "gru":
            self.actor = ActorGRU(args)
            self.critic = CriticGRU(args)
        else:
            raise ValueError("Invalid actor_critic_model")

        self.actor.to(self.device)
        self.critic.to(self.device)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)


    def _build_directory(self, args):
        base_path = "model/gpu/"
        if args.rl_type == "uncertainty":
            base_path += 'uncertainty/'
            if not (args.use_obs_imperfect or args.gather_to_some_region or args.use_beta_change or args.use_action_uncertainty):
                print("[WARN] rl_type=uncertainty but all uncertainty flags are False, please check")
            if args.use_obs_imperfect:
                base_path += 'obs_imperfect/'
            if args.gather_to_some_region:
                base_path += 'unsteady/'
            if args.use_beta_change or args.use_action_uncertainty:
                base_path += 'state_transition_uncertain/'

        base_path += args.actor_critic_model + '/'
        base_path = os.path.join(base_path, str(args.R0))

        if args.rl_type == "uncertainty" and args.use_obs_imperfect:
            base_path = os.path.join(base_path, args.predictor_type)

        details = f"{args.city}_{args.experiment_idx}"
        if args.state_contain_action:
            details += "_s_contain_a"
        details += f"_maxtrainsteps={args.max_train_steps}"

        path = os.path.join(base_path, details)

        return path

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        """Select action with highest probability for state s [B, window_size, zone_num, local_state_dim * 2]."""
        with torch.no_grad():
            # Get logits from actor network
            logits = self.actor(s) # [B, zone_num, action_dim]
            a = torch.argmax(logits, dim=-1) # [batch_size, zone_num]

        return a


    def choose_action(self, s):
        """Sample action from policy and return action, logprob."""

        with torch.no_grad():
            logits = self.actor(s)  # [B, zone_num, action_dim]
            probs = torch.softmax(logits, dim=-1)

            dist = Categorical(probs=probs)

            # Sample action [B, zone_num]
            action = dist.sample()

            # Calculate logprob
            logprob = dist.log_prob(action)  # [B, zone_num]

        return action, logprob


    def _calculate_gae(self, s, s_, r, dw, done):
        """Calculate Generalized Advantage Estimation (GAE)."""

        batch_size = s.size(0)
        mini_batch_size = 128  # Set mini batch size, adjust based on memory
        num_batches = (batch_size + mini_batch_size - 1) // mini_batch_size  # Calculate number of batches

        vs = torch.zeros_like(r)  # [B, zone_num]
        vs_ = torch.zeros_like(r)  # [B, zone_num]

        # Calculate vs and vs_ in batches
        with torch.no_grad():
            for i in range(num_batches):
                start = i * mini_batch_size
                end = min((i + 1) * mini_batch_size, batch_size)
                vs[start:end] = self.critic(s[start:end])  # Current state value
                vs_[start:end] = self.critic(s_[start:end])  # Next state value
        deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
        gae = torch.zeros(r.size(1), device = r.device)   # [zone_num]
        adv = torch.zeros_like(r)   # [B, zone_num]
        for i in reversed(range(r.size(0))):
            gae = deltas[i] + self.gamma * self.lamda * gae * (1.0 - done[i])
            adv[i] = gae # [zone_num]
        v_target = adv + vs
        if self.use_adv_norm:  # Normalize advantage if specified
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return adv, v_target

    def update(self, replay_buffer, total_steps):
        """Update actor-critic networks using PPO with GAE."""
        s, a, a_logprob, r, s_, dw, done = replay_buffer.get_stored_data()  # Get training data

        # Calculate GAE
        adv, v_target = self._calculate_gae(s, s_, r, dw, done)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                logits = self.actor(s[index])  # [mini_batch_size, zone_num, action_dim]
                probs = torch.softmax(logits, dim=-1)

                dist_now = Categorical(probs=probs)

                a_logprob_now = dist_now.log_prob(a[index, :])  # Current action log_prob
                dist_entropy = dist_now.entropy()  # [B, zone_num]
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size, zone_num)

                surr1 = ratios * adv[index]  # [B, zone_num]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                # Actor loss with entropy bonus for exploration
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # [B, zone_num]
                actor_loss = actor_loss.mean()

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index]) # [B, zone_num]
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            lr_now = self.lr_decay(total_steps)

        return -torch.min(surr1, surr2).mean(), critic_loss, dist_entropy.mean(), lr_now


    def lr_decay(self, total_steps):
        """Decay learning rate linearly."""
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        # Update learning rates
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
        # Entropy coefficient decay
        self.entropy_coef = self.entropy_coef_copy * (1 - total_steps / self.max_train_steps)
        return lr_a_now

    def save(self, episode):
        """Save model to disk."""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        torch.save(self.critic.state_dict(), os.path.join(self.directory, "ppo_critic{}.pth".format(episode)))
        torch.save(self.actor.state_dict(), os.path.join(self.directory, "ppo_actor{}.pth".format(episode)))
        print("Save RL model at path: ", os.path.join(self.directory, "ppo_actor{}.pth".format(episode)))

    def load(self, episode):
        """Load model from disk."""
        # Define file paths
        critic_path = os.path.join(self.directory, "ppo_critic{}.pth".format(episode))
        actor_path = os.path.join(self.directory, "ppo_actor{}.pth".format(episode))

        # Load based on device
        if self.device == torch.device("cpu"):
            # Load to CPU
            self.critic.load_state_dict(torch.load(critic_path, map_location=torch.device('cpu'), weights_only=True))
            self.actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            # Load to GPU
            self.critic.load_state_dict(torch.load(critic_path, weights_only=True))
            self.actor.load_state_dict(torch.load(actor_path, weights_only=True))
        print("Load RL model from path: ", actor_path)