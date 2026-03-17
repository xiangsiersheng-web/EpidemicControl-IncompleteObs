"""
对前两个特征（现存，新增）进行预测，同时使用动作作为特征
节点特征为：（现存，新增，u_p_test，u_p_quara）

模型重置：在 RebuildGruGNNModel 中，增加了模型参数重置的函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, bias=True):
        super(GraphConvolution, self).__init__()
        # MLP部分：单层线性变换
        self.mlp = nn.Linear(in_features, out_features, bias=bias)
        self.ln = nn.LayerNorm(out_features)
        # Dropout层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, OD, POP=None, obs_mask=None):
        """
        前向传播
        参数：
            x: 节点特征矩阵，形状为 (batch_size, num_nodes, in_features)
            OD: 行归一化的邻接矩阵，形状为 (num_nodes, num_nodes)
            POP: 人口数据，形状为 (num_nodes,)
            obs_mask: x中观测不到的区域，形状为 (batch_size, num_nodes)
        返回：
            聚合后的节点特征，形状为 (batch_size, num_nodes, out_features)
        *注意：
            目前的加权计算逻辑中，认为x未除以POP ！！！
        """
        OD = OD.unsqueeze(0).expand(x.shape[0], -1, -1)
        OD_t = OD.transpose(1, 2)

        # # 方式一：先MLP再聚合
        # x = self.mlp(x)
        # x = F.relu(x)
        # adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1)
        # adj_t = adj.transpose(1, 2)
        # # torch.bmm 期望输入为 (batch_size, m, n) @ (batch_size, n, p) -> (batch_size, m, p)
        # x = torch.bmm(adj_t, x)
        # x = torch.bmm(adj, x)
        # return x

        # 把x拆分为两部分，[curr_EI, new_EI], [u_p_test, u_p_quara]
        ei = x[:, :, :2]
        u = x[:, :, 2:]
        obs_mask = u[:, :, 0] == 0  # x中观测不到的区域，形状为 (batch_size, num_nodes)

        # 方式二：先聚合再MLP
        if POP is None:
            # torch.bmm 期望输入为 (batch_size, m, n) @ (batch_size, n, p) -> (batch_size, m, p)
            ei = torch.bmm(OD_t, ei)
            ei = torch.bmm(OD, ei)
        else:
            POP = POP.unsqueeze(0).expand(ei.shape[0], -1)   # (env.env_count, env.ZONE_NUM)
            predict = torch.bmm(OD_t, ei) / torch.bmm(OD_t, POP.unsqueeze(-1))    # (env.env_count, env.ZONE_NUM, 2)
            predict = torch.bmm(OD, predict) * POP.unsqueeze(-1)
            predict = torch.clip(predict, min=0)
            # 数值修正
            numerical_ratio = [(ei[i, ~obs_mask[i], :]).sum(dim=0) / (predict[i, ~obs_mask[i], :].sum(dim=0) + 1e-16)
                               for i in range(ei.shape[0])]
            numerical_ratio = torch.stack(numerical_ratio).unsqueeze(1) # (env.env_count, 1, node_features)
            predict = predict * numerical_ratio
            # ei[obs_mask] = predict[obs_mask]
            ei[obs_mask] = torch.max(predict[obs_mask], ei[obs_mask])

        x = torch.cat((ei, u), dim = -1) # (env.env_count, env.ZONE_NUM, 4)
        x = self.mlp(x)
        # 层归一化
        # x = self.ln(x)
        # 应用Dropout
        x = self.dropout(x)

        return F.relu(x)

    def reset_parameters(self, init_method='xavier'):
        """重置参数，支持多种初始化方法"""
        if init_method == 'xavier':
            nn.init.xavier_normal_(self.mlp.weight)
        elif init_method == 'he':
            nn.init.kaiming_normal_(self.mlp.weight, mode='fan_in', nonlinearity='relu')
        elif init_method == 'orthogonal':
            nn.init.orthogonal_(self.mlp.weight)
        else:
            raise ValueError(f"未知的初始化方法: {init_method}")

        if self.mlp.bias is not None:
            nn.init.zeros_(self.mlp.bias)



class RebuildGruGNNModel(nn.Module):
    """GRU-GNN 模型，继承自 nn.Module"""

    def __init__(self, gru_hidden_size, mlp_output_size,
                 gru_num_layers, env, device_name, node_output_size = 32):
        super(RebuildGruGNNModel, self).__init__()
        self.device = torch.device(device_name)
        self.env = env
        self.zone_num = env.ZONE_NUM
        self.adj = env.OD[0]
        self.POP = env.POP[0]

        # 定义 GNN 层
        node_feature_size = 4  #
        node_output_size = node_output_size  # 输出节点特征维度
        self.gnn = GraphConvolution(in_features=node_feature_size, out_features=node_output_size, bias=False)

        # 定义 GRU 层
        self.gru = nn.GRU(
            input_size=node_output_size,  # GNN 输出的维度
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bias=False
        )

        # 层归一化
        self.gru_layernorm = nn.LayerNorm(gru_hidden_size)

        # 定义输出层
        self.fc = nn.Linear(gru_hidden_size, mlp_output_size, bias=False)

    def forward(self, batch_inputs):
        """
        前向传播函数

        参数:
            batch_inputs (tuple): 输入数据，
                his_obs: (B, seq_len, zone_num, 2)
                actions: (B, seq_len, zone_num, 2), 表示 u_p_test, u_p_quara

        返回:
            predictions (torch.Tensor): 预测结果，形状为 (batch_size, output_size)
        """
        obs, actions = batch_inputs
        batch_size, seq_len, _, _ = obs.shape

        # 存储 GRU 的输入
        gru_inputs = []

        for t in range(seq_len):

            # 提取节点特征（感染人数和动作），形状为 (batch_size, zone_num, 4)
            i = obs[:, t, :, :]  # (batch_size, zone_num, 2)
            node_features = torch.cat((i, actions[:, t, :, :]), dim=-1)  # (batch_size, zone_num, 4)

            # 根据action确定此时的obs_mask:
            u_p_test = actions[:, t, :, 0]
            obs_mask = u_p_test == 0    # (batch_size, zone_num)

            # 通过 GNN 层提取特征
            gnn_outputs = self.gnn(node_features, self.adj, self.POP, obs_mask)  # (batch_size, zone_num, node_output_size)

            # 将 GNN 输出作为 GRU 的输入
            gru_inputs.append(gnn_outputs.unsqueeze(1))  # (batch_size, 1, zone_num, node_output_size)


        # 将所有时间步的输入拼接为 (batch_size, seq_len + 1, zone_num, node_output_size)
        gru_inputs = torch.cat(gru_inputs, dim=1)

        B, L, M, D = gru_inputs.shape
        gru_inputs = gru_inputs.permute(0, 2, 1, 3).reshape(B*M, L, D)

        # 前向传播通过 GRU
        outputs, _ = self.gru(gru_inputs)  # outputs: (B*M, seq_len, hidden_size)

        # 层归一化
        # outputs = self.gru_layernorm(outputs)

        # 取最后一个时间步的输出
        gru_last_output = outputs[:, -1, :]  # (batch_size, hidden_size)

        # 输出层
        predictions = self.fc(gru_last_output)  # (B*M, output_size)
        predictions = predictions.reshape(B, M, -1)

        # 使用 ReLU 激活函数
        predictions = F.relu(predictions)

        return predictions

    def reset_top_layer(self, init_method='xavier'):
        """
        重置最后的MLP层权重
        参数:
            init_method: 初始化方法 ('xavier', 'he', 'zeros')
        """
        if init_method == 'xavier':
            nn.init.xavier_normal_(self.fc.weight)
        elif init_method == 'he':
            nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        elif init_method == 'zeros':
            nn.init.zeros_(self.fc.weight)
        else:
            raise ValueError(f"未知的初始化方法: {init_method}")

        # 如果有偏置项也重置
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        print("MLP层权重已重置")

    def _reset_gru_layer(self, method='orthogonal'):
        """更安全的GRU层重置，默认使用正交初始化"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                if method == 'orthogonal':  # RNN类网络推荐正交初始化
                    nn.init.orthogonal_(param)
                elif method == 'xavier':
                    nn.init.xavier_normal_(param)
                elif method == 'he':
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                # 将遗忘门偏置初始化为1（改善梯度流动）
                if 'bias_hh' in name or 'bias_ih' in name:
                    n = param.size(0)
                    param.data.fill_(0)
                    param.data[n // 4:n // 2].fill_(1)  # 遗忘门偏置

    def reset_all_layers(self, init_method='xavier'):
        self.gnn.reset_parameters(init_method)

        # 重置gru层
        self._reset_gru_layer(method='orthogonal')

        self.reset_top_layer(init_method)

        print(f"所有层权重已用 {init_method} 方法重置（GRU层强制使用正交初始化）")


class RebuildGruGNN():
    """训练和评估 GRU-GNN 模型"""

    def __init__(self, model, device_name, seq_len, env):
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.seq_len = seq_len # 7
        self.env = env          # environment, 借用其中的 action_to_u 函数

        self.normalization_stats = {}

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _calculate_normalization_stats(self, true_state):
        # """计算均值和标准差（忽略零值）"""
        # # 对现存病例（第0维）
        # nonzero_mask_curr = true_state[..., 0] > 0
        # mu_curr = true_state[..., 0][nonzero_mask_curr].mean()
        # std_curr = true_state[..., 0][nonzero_mask_curr].std()
        #
        # # 对新增病例（第1维）
        # nonzero_mask_new = true_state[..., 1] > 0
        # mu_new = true_state[..., 1][nonzero_mask_new].mean()
        # std_new = true_state[..., 1][nonzero_mask_new].std()
        #
        # print("归一化参数 - 现存: μ={:.2f}, σ={:.2f} | 新增: μ={:.2f}, σ={:.2f}".format(
        #     mu_curr, std_curr,
        #     mu_new, std_new
        # ))
        #
        # return {
        #     'mu_curr': mu_curr, 'std_curr': std_curr,
        #     'mu_new': mu_new, 'std_new': std_new
        # }

        max_curr = torch.max(true_state[..., 0])
        max_new = torch.max(true_state[..., 1])

        print("归一化参数 - 现存: max={:.2f} | 新增: max={:.2f}".format(max_curr, max_new))

        return {
            'max_curr': max_curr, 'max_new': max_new
        }

    def _normalize(self, x, feature_dim):
        # """应用Z-Score归一化"""
        # if feature_dim == 0:  # 现存病例
        #     return (x - self.normalization_stats['mu_curr']) / (self.normalization_stats['std_curr'] + 1e-8)
        # else:  # 新增病例
        #     return (x - self.normalization_stats['mu_new']) / (self.normalization_stats['std_new'] + 1e-8)

        if feature_dim == 0:  # 现存病例
            return x / (self.normalization_stats['max_curr'] + 1e-8)
        else:  # 新增病例
            return x / (self.normalization_stats['max_new'] + 1e-8)

    def _denormalize(self, x, feature_dim):
        # """反归一化"""
        # if feature_dim == 0:  # 现存病例
        #     return x * self.normalization_stats['std_curr'] + self.normalization_stats['mu_curr']
        # else:  # 新增病例
        #     return x * self.normalization_stats['std_new'] + self.normalization_stats['mu_new']

        if feature_dim == 0:  # 现存病例
            return x * (self.normalization_stats['max_curr'] + 1e-8)
        else:  # 新增病例
            return x * (self.normalization_stats['max_new'] + 1e-8)

    def train(self, dataset, num_epochs=20, batch_size=64, validation_split=0.05, patience=5):
        """使用数据集训练网络"""
        # 准备数据加载器
        inputs, targets = self.prepare_dataset(dataset) # inputs: tuple of (obs, actions)
        dataset = TensorDataset(*inputs, targets)

        # 划分训练集和验证集
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 先评估一次
        val_avg_loss, correct_ratio = self.validate(val_loader)
        print(f'Epoch [0], Validation Loss: {val_avg_loss:.6f}, Validation Correct Ratio: {correct_ratio:.4f}')

        # 早停机制参数
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False
        best_model_state = None

        for epoch in range(num_epochs):
            if early_stop:
                print("早停触发，停止训练")
                break

            self.model.train()
            epoch_loss = 0.0
            for batch_obs, batch_actions, batch_targets in train_loader:
                batch_obs = batch_obs.to(self.device)  # (batch_size, seq_len, zone_num, 2)
                batch_actions = batch_actions.to(self.device)  # (batch_size, seq_len, zone_num, 2)
                batch_inputs = (batch_obs, batch_actions)
                batch_targets = batch_targets.to(self.device)  # (batch_size, output_size)

                # 前向传播
                predictions = self.model(batch_inputs)

                # 计算损失
                loss = self.criterion(predictions, batch_targets)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.6f}')

            # 在验证集上评估模型
            val_avg_loss, correct_ratio = self.validate(val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_avg_loss:.6f}, Validation Correct Ratio: {correct_ratio:.4f}')

            # 早停机制判断
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()  # 保存当前最佳模型
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"验证损失在 {patience} 个 epoch 中没有改善，停止训练。")
                    early_stop = True

            # 训练结束后，加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("加载验证集上性能最好的模型。")

    def validate(self, val_loader):
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch_obs, batch_actions, batch_targets in val_loader:
                batch_obs = batch_obs.to(self.device)  # (batch_size, seq_len, zone_num, 2)
                batch_actions = batch_actions.to(self.device)  # (batch_size, seq_len, zone_num, 2)
                batch_inputs = (batch_obs, batch_actions)
                batch_targets = batch_targets.to(self.device)  # (batch_size, output_size)

                predictions = self.model(batch_inputs)
                loss = self.criterion(predictions, batch_targets)
                total_loss += loss.item()

                # 计算相对误差
                abs_error = torch.abs(predictions - batch_targets)  # (batch_size, zone_num)

                # 处理实际值为0的情况
                relative_error = torch.where(
                    batch_targets != 0,
                    abs_error / torch.abs(batch_targets),
                    torch.where(predictions == 0, torch.zeros_like(abs_error),
                                torch.ones_like(abs_error) * float('inf'))
                )

                # 判断相对误差是否小于30%
                is_correct = relative_error < 0.3  # (batch_size, zone_num)

                # 统计正确的预测数
                correct_predictions += is_correct.sum().item()
                total_samples += batch_targets.numel()  # batch_size * zone_num

        avg_loss = total_loss / len(val_loader)
        accuracy_rate = (correct_predictions / total_samples) * 100  # 百分比
        return avg_loss, accuracy_rate

    def predict(self, obs, actions):
        """给定输入，输出预测值
        :param obs: 观测，shape=(batch_size, seq_len, zone_num, 2)
        :param actions: 动作，shape=(batch_size, seq_len, zone_num)

        :return prediction: 预测值，shape=(batch_size, zone_num, 2)
        """
        self.model.eval()
        obs = obs.clone()

        obs0 = obs[:, :, :, :2]     # (batch_size, seq_len, zone_num, 2)
        obs0[:, :, :, 0] = self._normalize(obs0[:, :, :, 0], 0)
        obs0[:, :, :, 1] = self._normalize(obs0[:, :, :, 1], 1)

        u_p_test, u_p_quara = self.env._action_to_u(actions)
        actions0 = torch.stack((u_p_test, u_p_quara), dim=-1)   # (batch_size, seq_len, zone_num, 2)

        with torch.no_grad():
            # 预处理输入数据
            inputs = (obs0, actions0)

            # 前向传播
            predictions = self.model(inputs)

            # 反归一化
            predictions[:, :, 0] = self._denormalize(predictions[:, :, 0], 0)
            predictions[:, :, 1] = self._denormalize(predictions[:, :, 1], 1)

            # 确保预测值非负
            predictions = torch.clamp(predictions, min=0)

        return predictions  # 返回预测结果

    def load(self, model_path):
        """加载模型文件"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 加载归一化参数
        if 'normalization_stats' in checkpoint:
            self.normalization_stats = checkpoint['normalization_stats']
            print("DL 成功加载归一化参数：", model_path)
        else:
            print("警告：未找到归一化参数，预测结果可能不准确！")

    def save(self, model_path):
        """保存模型文件"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalization_stats': self.normalization_stats
        }, model_path)

    def prepare_dataset(self, dataset):
        """准备训练数据集的输入和目标张量，使用滑动窗口方式"""
        obs = dataset['imperfect_obs'][:,:,:,:2]        # (num_samples, total_days, zone_num, 2)
        actions = dataset['history_action']   # (num_samples, total_days, zone_num)
        true_state = dataset['true_state'][:,:,:,:2]     # (num_samples, total_days, zone_num, 2)
        print("obs.shape:", obs.shape, "\t actions.shape:", actions.shape, "\t true_state.shape:", true_state.shape)

        # # # TODO: 对obs，true_state进行最大最小归一化
        # self.max_curr = torch.max(true_state[:, :, :, 0])
        # self.max_new = torch.max(true_state[:, :, :, 1])
        # true_state[:, :, :, 0] /= self.max_curr
        # true_state[:, :, :, 1] /= self.max_new
        # obs[:, :, :, 0] /= self.max_curr
        # obs[:, :, :, 1] /= self.max_new

        # 计算并保存归一化参数（基于真实数据）
        self.normalization_stats = self._calculate_normalization_stats(true_state)

        num_samples, total_days, zone_num, obs_dim = obs.shape

        # 在各数据前填充seq_len-1个全零的观测和动作
        obs = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num, obs_dim), device=obs.device), obs], dim=1)
        actions = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num), device=obs.device), actions], dim=1)
        true_state = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num, obs_dim), device=obs.device), true_state], dim=1)

        # 应用归一化
        true_state[:, :, :, 0] = self._normalize(true_state[:, :, :, 0], 0)
        true_state[:, :, :, 1] = self._normalize(true_state[:, :, :, 1], 1)
        obs[:, :, :, 0] = self._normalize(obs[:, :, :, 0], 0)
        obs[:, :, :, 1] = self._normalize(obs[:, :, :, 1], 1)

        num_samples, total_days, zone_num, obs_dim = obs.shape

        inputs_obs = []
        inputs_actions = []
        targets = []

        for i in range(num_samples):
            for day in range(self.seq_len, total_days):  # 注意这里的索引调整，跳过第0天
                # 处理观测数据
                obs_seq = obs[i, day - self.seq_len + 1 : day + 1, :]  # (seq_len, zone_num, 2)

                # 处理动作数据（由于 s_t, a_t, s_t+1 的理解，对s_t+1产生影响的动作存储在t天
                action_seq = actions[i, day - self.seq_len : day, :]  # (seq_len, zone_num)
                # 将动作映射为检测隔离率
                u_p_test, u_p_quara = self.env._action_to_u(action_seq)

                # 获取当天的真实状态
                target = true_state[i, day, :]  # (zone_num)

                inputs_obs.append(obs_seq)
                inputs_actions.append(torch.stack((u_p_test, u_p_quara), dim=-1))
                targets.append(target)

        # 将列表转换为张量
        inputs_obs = torch.stack(inputs_obs)                # (total_samples, seq_len, zone_num, 2)
        inputs_actions = torch.stack(inputs_actions)        # (total_samples, seq_len, zone_num, 2)
        targets = torch.stack(targets)                      # (total_samples, zone_num)
        print("inputs_obs.shape:", inputs_obs.shape, "\t inputs_actions.shape:", inputs_actions.shape, "\t targets.shape:", targets.shape)

        return (inputs_obs, inputs_actions), targets
