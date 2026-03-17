# 只对第一个特征（现存）进行预测，不使用动作作为特征
# 节点特征为：（现存）

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

from environment.uncertain_seir_vector_v4 import EpidemicModel


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, bias=True):
        super(GraphConvolution, self).__init__()
        # MLP部分：单层线性变换
        self.mlp = nn.Linear(in_features, out_features, bias=bias)
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
        注意：
            目前的加权认为x未除以POP
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

        # 方式二：先聚合再MLP
        if POP is None:
            # torch.bmm 期望输入为 (batch_size, m, n) @ (batch_size, n, p) -> (batch_size, m, p)
            x = torch.bmm(OD_t, x)
            x = torch.bmm(OD, x)
            x = self.mlp(x)
            # 应用Dropout
            x = self.dropout(x)
        else:
            POP = POP.unsqueeze(0).expand(x.shape[0], -1)   # (env.env_count, env.ZONE_NUM)
            predict = torch.bmm(OD_t, x) / torch.bmm(OD_t, POP.unsqueeze(-1))    # (env.env_count, env.ZONE_NUM, 2)
            predict = torch.bmm(OD, predict) * POP.unsqueeze(-1)
            predict = torch.clip(predict, min=0)
            # 数值修正
            numerical_ratio = [(x[i, ~obs_mask[i], :]).sum(dim=0) / (predict[i, ~obs_mask[i], :].sum(dim=0) + 1e-16)
                               for i in range(x.shape[0])]
            numerical_ratio = torch.stack(numerical_ratio).unsqueeze(1) # (env.env_count, 1, node_features)
            predict = predict * numerical_ratio
            # x[obs_mask] = predict[obs_mask]
            x[obs_mask] = torch.max(predict[obs_mask], x[obs_mask])

            x = self.mlp(x)
            # 应用Dropout
            x = self.dropout(x)

        return F.relu(x)



class RebuildGruGNNModel(nn.Module):
    """GRU-GNN 模型，继承自 nn.Module"""

    def __init__(self, gru_hidden_size, mlp_output_size, gru_num_layers, env, device_name):
        super(RebuildGruGNNModel, self).__init__()
        self.device = torch.device(device_name)
        self.env = env
        self.zone_num = env.ZONE_NUM
        self.adj = env.OD[0]
        self.POP = env.POP[0]

        # 定义 GNN 层
        node_feature_size = 1  #
        node_output_size = 32  # 输出节点特征维度
        self.gnn = GraphConvolution(in_features=node_feature_size, out_features=node_output_size, bias=False)

        # 定义 GRU 层
        self.gru = nn.GRU(
            input_size=node_output_size,  # GNN 输出的维度
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bias=False
        )

        # 定义输出层
        # 将 GRU 的 hidden_size 和 zone_num 拼接后输入到全连接层
        self.fc = nn.Linear(gru_hidden_size, mlp_output_size, bias=False)

    def forward(self, batch_inputs):
        """
        前向传播函数

        参数:
            batch_inputs (tuple): 输入数据，
                his_obs: (B, seq_len, zone_num, 2)
                actions: (B, seq_len, zone_num)
                curr_obs: (B, zone_num, 2)

        返回:
            predictions (torch.Tensor): 预测结果，形状为 (batch_size, output_size)
        """
        obs, actions = batch_inputs
        batch_size, seq_len, _, _ = obs.shape

        # 存储 GRU 的输入
        gru_inputs = []

        for t in range(seq_len):

            # 提取节点特征（感染人数和动作），形状为 (batch_size, zone_num, 2)
            i = obs[:, t, :, :]  # (batch_size, zone_num, 2)
            node_features = i  # (batch_size, zone_num, 2)

            # 根据action确定此时的obs_mask:
            u_p_test, u_p_quara = self.env._action_to_u(actions[:, t, :])
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

        # 取最后一个时间步的输出
        gru_last_output = outputs[:, -1, :]  # (batch_size, hidden_size)

        # 输出层
        predictions = self.fc(gru_last_output)  # (B*M, output_size)
        predictions = predictions.reshape(B, M, -1)

        # 使用 ReLU 激活函数
        predictions = F.relu(predictions)

        return predictions


class RebuildGruGNN():
    """训练和评估 GRU-GNN 模型"""

    def __init__(self, model, device_name, seq_len):
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.seq_len = seq_len # 7

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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
                batch_obs = batch_obs.to(self.device)  # (batch_size, seq_len, zone_num)
                batch_actions = batch_actions.to(self.device)  # (batch_size, seq_len, zone_num)
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
                batch_obs = batch_obs.to(self.device)  # (batch_size, seq_len, zone_num)
                batch_actions = batch_actions.to(self.device)  # (batch_size, seq_len, zone_num)
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
        """
        self.model.eval()
        obs = obs.clone()

        obs0 = obs[:, :, :, 0].unsqueeze(-1) # (batch_size, seq_len, zone_num, 1)

        with torch.no_grad():
            # 预处理输入数据
            inputs = (obs0, actions)

            # 前向传播
            predictions = self.model(inputs)

        # return predictions  # 返回预测结果
        predictions = torch.cat((predictions, obs[:, -1, :, 1].unsqueeze(-1)),dim = -1)
        return predictions

    def load(self, model_path):
        """加载模型文件"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self, model_path):
        """保存模型文件"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

    def prepare_dataset(self, dataset):
        """准备训练数据集的输入和目标张量，使用滑动窗口方式"""
        obs = dataset['imperfect_obs'][:,:,:,0].unsqueeze(-1)        # (num_samples, total_days, zone_num, 1)
        actions = dataset['history_action']   # (num_samples, total_days, zone_num)
        true_state = dataset['true_state'][:,:,:,0].unsqueeze(-1)     # (num_samples, total_days, zone_num, 1)
        print("obs.shape:", obs.shape, "\t actions.shape:", actions.shape, "\t true_state.shape:", true_state.shape)

        # # TODO: 对obs，true_state进行最大最小归一化

        num_samples, total_days, zone_num, obs_dim = obs.shape

        # 在各数据前填充seq_len-1个全零的观测和动作
        obs = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num, obs_dim), device=obs.device), obs], dim=1)
        actions = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num), device=obs.device), actions], dim=1)
        true_state = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num, obs_dim), device=obs.device), true_state], dim=1)

        num_samples, total_days, zone_num, obs_dim = obs.shape

        inputs_obs = []
        inputs_actions = []
        targets = []

        for i in range(num_samples):
            for day in range(self.seq_len, total_days):  # 注意这里的索引调整，跳过第0天
                # 处理观测数据
                obs_seq = obs[i, day - self.seq_len + 1 : day + 1, :]  # (seq_len, zone_num)

                # 处理动作数据（由于 s_t, a_t, s_t+1 的理解，对s_t+1产生影响的动作存储在t天
                action_seq = actions[i, day - self.seq_len : day, :]  # (seq_len, zone_num)

                # 获取当天的真实状态
                target = true_state[i, day, :]  # (zone_num)

                inputs_obs.append(obs_seq)
                inputs_actions.append(action_seq)
                targets.append(target)

        # 将列表转换为张量
        inputs_obs = torch.stack(inputs_obs)                # (total_samples, seq_len, zone_num)
        inputs_actions = torch.stack(inputs_actions)        # (total_samples, seq_len, zone_num)
        targets = torch.stack(targets)                      # (total_samples, zone_num)
        print("inputs_obs.shape:", inputs_obs.shape, "\t inputs_actions.shape:", inputs_actions.shape, "\t targets.shape:", targets.shape)

        return (inputs_obs, inputs_actions), targets

    def min_max_normalization(self, data):
        pass
