import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# 定义图神经网络模型
class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 加载数据集
dataset = Planetoid(root='/path/to/dataset', name='Cora')
data = dataset[0]

# 创建图神经网络模型
model = GNNModel(dataset.num_features, 16, dataset.num_classes)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# 模型训练
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 模型评估
def evaluate():
    model.eval()
    output = model(data.x, data.edge_index)
    pred = output.argmax(dim=1)
    acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

# 迭代训练和评估
for epoch in range(200):
    train()
    acc = evaluate()
    print(f'Epoch: {epoch+1}, Test Accuracy: {acc:.4f}')
