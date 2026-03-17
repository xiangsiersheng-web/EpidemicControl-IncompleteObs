import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCNModel(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, dropout=0.5):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_features)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # 图卷积层1
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # 图卷积层2
        x = self.conv2(x, edge_index, edge_weight)
        return x


# 创建一个简单的示例数据
N = 600  # 节点数量
in_features = 2  # 输入特征维度 (newI, currI)
hidden_dim = 64  # 隐藏层特征维度
out_features = 2  # 输出特征维度 (newI, currI)
dropout = 0.5  # dropout

# 假设你已经有了一个行归一化后的OD矩阵（邻接矩阵）
# 这里假设我们生成一个随机的行归一化OD矩阵作为邻接矩阵
OD_matrix = torch.rand(N, N)
OD_matrix = OD_matrix / OD_matrix.sum(dim=1, keepdim=True)  # 行归一化

# 将OD矩阵转换为PyTorch Geometric格式的邻接矩阵（edge_index）
edge_index = OD_matrix.nonzero(as_tuple=False).t()  # 找到非零元素的索引
edge_weight = OD_matrix[edge_index[0], edge_index[1]]  # 非零元素的值作为边的权重

# 创建目标矩阵 y，其中前k个节点是真实数据，后k个节点为缺失数据（即0）
y = torch.rand(N, in_features)  # 目标矩阵

# 创建一个节点特征矩阵 X，其中一部分节点的信息是已知的，另一部分是缺失的
k = 300  # 假设300个节点有真实数据
X = torch.zeros(N, in_features)  # 初始化节点特征矩阵

# 假设前k个节点的特征是已知的，随机生成真实数据
X[:k, 0] = y[:k, 0].clone()  # newI
X[:k, 1] = y[:k, 1].clone()  # currI

# 创建掩码（mask）来标记哪些节点的特征是已知的，哪些是未知的
mask_known = torch.zeros(N, dtype=torch.bool)
mask_known[:k] = 1  # 标记前k个节点为已知节点

# 创建图数据对象
data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight)

# 初始化模型
model = GCNModel(in_features=in_features, hidden_dim=hidden_dim, out_features=out_features, dropout=dropout)


# 使用均方误差（MSE）损失函数来训练模型
def loss_fn(pred, target, mask_known):
    # 对所有节点计算损失，损失函数包含已知节点和未知节点
    loss = F.mse_loss(pred, target)  # 对所有节点的损失进行计算
    return loss


# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(model, data, target, mask_known, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)  # 前向传播

        # 计算损失，包含所有节点的损失
        loss = loss_fn(out, target, mask_known)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


# 开始训练
train(model, data, y, mask_known)

# 训练完成后，进行推断
model.eval()
with torch.no_grad():
    predictions = model(data.x, data.edge_index, data.edge_attr)
    print(predictions)  # 输出所有节点的预测特征，包括未知节点的预测值
    print(torch.abs((predictions - y)/y))
