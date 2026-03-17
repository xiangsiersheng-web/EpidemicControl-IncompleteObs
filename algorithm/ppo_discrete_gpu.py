# -*- ecoding: utf-8 -*-
# @ModuleName: ppo_discrete
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
        """
        根据args初始化，actor critic网络结构，优化器，学习率等参数
        """
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

        # 构建模型保存路径
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
                print("[WARN] ppo_discrete_gpu.py: 要使用非确定类型的RL，但四类不确定性都为false，请检查是否符合预期")
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
        """
        根据当前状态s([B, window_sim, zone_num, local_state_dim * 2])，使用actor网络，为每个状态选择最大概率的动作。
        `batch_size` 指的是输入的s的第一个维度，不是这个函数里的batch_size
        """
        with torch.no_grad():
            # 通过actor网络获取logits
            logits = self.actor(s) # [B, zone_num, action_dim]
            a = torch.argmax(logits, dim=-1) # [batch_size, zone_num]

        return a


    def choose_action(self, s):
        """
        根据当前状态s([B, window_sim, zone_num, local_state_dim * 2])，概率抽样选择动作，并计算logprob和entropy
        `batch_size` 指的是输入的s的第一个维度，不是这个函数里的batch_size
        return : action, logprob, entropy
        """

        with torch.no_grad():
            logits = self.actor(s)  # [B, zone_num, action_dim]
            probs = torch.softmax(logits, dim=-1)

            dist = Categorical(probs=probs)

            # 抽样动作，形状为 [B, zone_num]
            action = dist.sample()

            # 计算 logprob
            logprob = dist.log_prob(action)  # [B, zone_num]

        return action, logprob


    def _calculate_gae(self, s, s_, r, dw, done):
        """
        计算广义优势估计 (GAE)

        Calculate the advantage using GAE
        'done=True' means dead or win, there is no next state s'
        'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0


        :return: 广义优势估计值和目标价值函数
        """

        batch_size = s.size(0)
        mini_batch_size = 128  # 设置小批量大小，依据内存情况调整
        num_batches = (batch_size + mini_batch_size - 1) // mini_batch_size  # 计算分批次数

        vs = torch.zeros_like(r)  # [B, zone_num]
        vs_ = torch.zeros_like(r)  # [B, zone_num]

        # 分批计算 vs 和 vs_
        with torch.no_grad():  # 关闭梯度计算
            for i in range(num_batches):
                start = i * mini_batch_size
                end = min((i + 1) * mini_batch_size, batch_size)
                vs[start:end] = self.critic(s[start:end])  # 当前状态价值
                vs_[start:end] = self.critic(s_[start:end])  # 下一状态价值

        # 这里应该使用dw，对于疫情控制，没有哪个状态是终止的，每个状态都有下一状态s_
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
        s, a, a_logprob, r, s_, dw, done = replay_buffer.get_stored_data()  # Get training data

        # 计算 GAE，是基于 critic 网络计算的
        adv, v_target = self._calculate_gae(s, s_, r, dw, done) # 计算广义优势估计, adv [self.batch_size, 1] v_target [self.batch_size, 1]

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                ## 首次进入该循环时，actor网络未更新，但之后的循环，actor是已经更新过的了
                logits = self.actor(s[index])  # [mini_batch_size, zone_num, action_dim]
                probs = torch.softmax(logits, dim=-1)

                dist_now = Categorical(probs=probs)

                a_logprob_now = dist_now.log_prob(a[index, :])  # [B, zone_num] # 当前动作 log_prob
                dist_entropy = dist_now.entropy()  # [B, zone_num]
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size, zone_num)

                surr1 = ratios * adv[index]  # [B, zone_num]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                # actor损失函数考虑了最小化负的熵，也即最大化熵，可以增加探索（最大化熵，会使动作倾向于均匀分布）
                # TODO： 这个熵的系数是否也应该衰减？
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # [B, zone_num]
                actor_loss = actor_loss.mean()

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 梯度范数不大于0.5
                self.optimizer_actor.step()

                v_s = self.critic(s[index]) # [B, zone_num]
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
        # 更新每个 actor 的学习率
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
        # # 熵系数衰减：
        self.entropy_coef = self.entropy_coef_copy * (1 - total_steps / self.max_train_steps)
        return lr_a_now

    def save(self, episode):
        """保存模型"""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        torch.save(self.critic.state_dict(), os.path.join(self.directory, "ppo_critic{}.pth".format(episode)))
        torch.save(self.actor.state_dict(), os.path.join(self.directory, "ppo_actor{}.pth".format(episode)))
        print("Save RL model at path: ", os.path.join(self.directory, "ppo_actor{}.pth".format(episode)))

    def load(self, episode):
        """加载模型"""
        # 定义文件路径
        critic_path = os.path.join(self.directory, "ppo_critic{}.pth".format(episode))
        actor_path = os.path.join(self.directory, "ppo_actor{}.pth".format(episode))

        # 根据设备选择加载方式
        if self.device == torch.device("cpu"):
            # 如果设备是 CPU，将模型加载到 CPU
            self.critic.load_state_dict(torch.load(critic_path, map_location=torch.device('cpu'), weights_only=True))
            self.actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            # 否则加载到 CUDA（GPU）
            self.critic.load_state_dict(torch.load(critic_path, weights_only=True))
            self.actor.load_state_dict(torch.load(actor_path, weights_only=True))
        print("Load RL model from path: ", actor_path)