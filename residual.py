import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 生成简单数据集
torch.manual_seed(0)
X = torch.randn(1000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(1000, 1)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 普通 MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ResNet Block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out + identity)

# ResNet MLP
class ResNetMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(20, 64)
        self.blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.blocks(x)
        return self.output(x)

# 训练函数
def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(100):
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(loader))
    return losses

# 开始训练
mlp_loss = train(SimpleMLP())
resnet_loss = train(ResNetMLP())

# 绘制对比图
plt.plot(mlp_loss, label="Simple MLP")
plt.plot(resnet_loss, label="ResNet MLP")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss: MLP vs ResNet MLP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
