# -*- ecoding: utf-8 -*-
# @ModuleName: ppo_discrete
# @Function: 
#  
# @Time: 2023/10/30 14:20
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import os

from dynamic.env_change_rules.beta_change import BetaChangeRule


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module): 
    """
    初始化actor网络结构。
    - 输入层：输入状态向量，形状为(state_dim * zone_num, 1)
    - 隐藏层fc1：hidden_width个神经元，输出形状为(hidden_width, 1)
    - 隐藏层fc2：hidden_width个神经元，输出形状为(hidden_width, 1)
    - 多头输出层heads：每个头输出动作概率分布，输出形状为(action_dim, 1)

    前向传播：通过网络接收输入状态向量s，输出各区域动作概率分布。
    """   
    def __init__(self, args):        
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.state_dim * args.zone_num, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.zone_num = args.zone_num

        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(args.hidden_width, args.action_dim) for _ in range(self.zone_num)
        ])

        # self.fc3 = nn.Linear(args.hidden_width, args.action_dim * args.zone_num)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            # print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            for head in self.heads:
                orthogonal_init(head, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        logits  = [self.heads[i](s) for i in range(self.zone_num)]

        # logits = self.fc3(s)

        return logits


class Critic(nn.Module):
    """
    初始化critic网络结构。
    - 输入层：接收状态向量，形状为(state_dim * zone_num, 1)
    - 隐藏层fc1：包含hidden_width个神经元，输出形状为(hidden_width, 1)
    - 隐藏层fc2：包含hidden_width个神经元，输出形状为(hidden_width, 1)
    - 输出层fc3：输出单个值，代表状态的价值估计

    前向传播：该函数接收状态s，通过网络计算并返回状态的价值。
    """
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim * args.zone_num, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            # print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_discrete:
    def __init__(self, args):
        """
        根据args初始化，actor critic网络结构，优化器，学习率等参数
        """
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
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.action_dim = args.action_dim
        self.state_dim = args.state_dim
        self.zone_num = args.zone_num

        if args.use_obs_noise:
            self.directory = "./model/{}/{}_{}/{}".format("noise", args.city, args.experiment_idx, args.obs_noise_std)
        elif args.beta_change_rule != BetaChangeRule.NONE:
            self.directory = "./model/{}/{}_{}/{}".format("dynamic_env", args.city, args.experiment_idx, "betachange=" + args.beta_change_rule.value)
            self.directory += args.timenow # 因为beta的变化难以表述，为了区分，就加上了时间戳
        else:
            self.directory = "./model/{}/{}_{}".format(args.R0, args.city, args.experiment_idx)
        if args.max_train_steps != 1.2e6:
            self.directory += "_maxtrainsteps=" + str(args.max_train_steps)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.actor = Actor(args)
        self.critic = Critic(args)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        """
        根据当前状态s([1, 518])，使用actor网络，选择最大概率的动作
        """
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        logits = self.actor(s)
        split_logits = logits
        # split_logits = torch.split(logits, [self.action_dim] * self.zone_num, dim=-1)
        action = torch.Tensor([torch.argmax(logits).item() for logits in split_logits])
        # 这里的action转置前是(74,)
        return action.T

    def choose_action(self, s):
        """
        根据当前状态s([1, 518])，概率抽样选择动作，并计算logprob和entropy
        """
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            logits = self.actor(s)
            split_logits = logits
            # split_logits = torch.split(logits, [self.action_dim] * self.zone_num, dim=-1)
            multi_categoricals = [Categorical(probs=torch.softmax(logits, dim=-1)) for logits in split_logits]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        # 这里的action转置前是(74,1)
        # 将其改为(74,)
        action = action[:, 0]
        return action.T, logprob.sum(0), entropy.sum(0)

    def calculate_gae(self, s, s_, r, dw, done):
        """
        计算广义优势估计 (GAE)

        Calculate the advantage using GAE
        'done=True' means dead or win, there is no next state s'
        'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0


        :return: 广义优势估计值和目标价值函数
        """
        # adv = []
        # gae = 0
        # with torch.no_grad():  # adv and v_target have no gradient
        #     vs = self.critic(s)
        #     vs_ = self.critic(s_)
        #     deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
        #     for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
        #         gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
        #         adv.insert(0, gae)
        #     adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
        #     v_target = adv + vs
        #     if self.use_adv_norm:  # Trick 1:advantage normalization
        #         adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        #
        # return adv, v_target

        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            gae = 0
            adv = torch.zeros_like(r)
            for i in reversed(range(len(r))):
                gae = deltas[i] + self.gamma * self.lamda * gae * (1.0 - done[i])
                adv[i] = gae
            v_target = adv + vs
            if self.use_adv_norm:  # Normalize advantage if specified
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        return adv, v_target

    def update(self, replay_buffer, total_steps):
        """
        根据replay_buffer数据，更新actor critic网络

        此函数用于：

        - 训练actor critic网络
            - 使用广义优势估计计算优势函数 `adv`
            - 使用`adv`计算 `v_target`
        - 使用PPO算法
            - 使用截断概率裁剪，计算`actor_loss`
            - 计算`actor_loss`时，加入了 最大化熵的目标函数，以增加探索
        - 使用了梯度截断，防止梯度爆炸
        - 使用了学习率衰减

        返回：
        - -actor_loss: 负的actor网络的loss
        - critic_loss: critic网络的loss
        - dist_entropy.mean()：动作的概率熵
        - lr_now: 当前actor网络学习率
        """
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data

        # 计算 GAE，是基于 critic 网络计算的
        adv, v_target = self.calculate_gae(s, s_, r, dw, done)

        # adv = []
        # gae = 0
        # with torch.no_grad():  # adv and v_target have no gradient
        #     vs = self.critic(s)
        #     vs_ = self.critic(s_)
        #     deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
        #     for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
        #         gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
        #         adv.insert(0, gae)
        #     adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
        #     v_target = adv + vs
        #     if self.use_adv_norm:  # Trick 1:advantage normalization
        #         adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                ## 这时的actor网络应该还未更新，新算出的probs和存储的probs应该是一样的吧？
                ## 首次进入该循环时，actor网络未更新，但之后的循环，actor是已经更新过的了
                probs = self.actor(s[index])
                split_probs=probs
                # split_probs = torch.split(probs, [self.action_dim] * self.zone_num, dim=-1)

                dist_now = [Categorical(probs=torch.softmax(prob, dim=-1)) for prob in split_probs]

                a_logprob_now = torch.stack(
                    [categorical.log_prob(a) for a, categorical in zip(a[index].T, dist_now)]).sum(0).view(-1, 1)
                dist_entropy = torch.stack([categorical.entropy() for categorical in dist_now]).sum(0).view(-1, 1)

                # a/b=exp(log(a)-log(b))

                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                # actor损失函数考虑了最小化负的熵，也即最大化熵，可以增加探索（最大化熵，会使动作倾向于均匀分布）
                # TODO： 这个熵的系数是否也应该衰减？
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 梯度范数不大于0.5
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s) # 均方误差作为损失函数
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
        """学习率衰减"""
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)

        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
        return lr_a_now

    def save(self, episode):
        """保存模型"""
        torch.save(self.critic.state_dict(), os.path.join(self.directory, "ppo_critic{}.pth".format(episode)))
        torch.save(self.actor.state_dict(), os.path.join(self.directory, "ppo_actor{}.pth".format(episode)))

    def load(self, episode):
        """加载模型"""
        self.critic.load_state_dict(torch.load(os.path.join(self.directory, "ppo_critic{}.pth".format(episode))))
        self.actor.load_state_dict(torch.load(os.path.join(self.directory, "ppo_actor{}.pth".format(episode))))

