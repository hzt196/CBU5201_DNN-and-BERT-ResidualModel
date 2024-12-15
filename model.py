import torch
import torch.nn as nn

class ResidualClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(ResidualClassifier, self).__init__()
        # 定义12层的线性层
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layer6 = nn.Linear(hidden_size, hidden_size)
        self.layer7 = nn.Linear(hidden_size, hidden_size)
        self.layer8 = nn.Linear(hidden_size, hidden_size)
        self.layer9 = nn.Linear(hidden_size, hidden_size)
        self.layer10 = nn.Linear(hidden_size, hidden_size)
        self.layer11 = nn.Linear(hidden_size, hidden_size)
        self.layer12 = nn.Linear(hidden_size, 2)  # 输出为2，适合二分类

        # 定义对应的Batch Normalization层
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.bn7 = nn.BatchNorm1d(hidden_size)
        self.bn8 = nn.BatchNorm1d(hidden_size)
        self.bn9 = nn.BatchNorm1d(hidden_size)
        self.bn10 = nn.BatchNorm1d(hidden_size)
        self.bn11 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # 第一个层没有残差连接
        x = torch.relu(self.bn1(self.layer1(x)))

        # 每两个线性层之间增加残差连接
        residual = x
        x = torch.relu(self.bn2(self.layer2(x)) + residual)

        residual = x
        x = torch.relu(self.bn3(self.layer3(x)) + residual)

        residual = x
        x = torch.relu(self.bn4(self.layer4(x)) + residual)

        residual = x
        x = torch.relu(self.bn5(self.layer5(x)) + residual)

        residual = x
        x = torch.relu(self.bn6(self.layer6(x)) + residual)

        residual = x
        x = torch.relu(self.bn7(self.layer7(x)) + residual)

        residual = x
        x = torch.relu(self.bn8(self.layer8(x)) + residual)

        residual = x
        x = torch.relu(self.bn9(self.layer9(x)) + residual)

        residual = x
        x = torch.relu(self.bn10(self.layer10(x)) + residual)

        residual = x
        x = torch.relu(self.bn11(self.layer11(x)) + residual)

        # 输出层，无残差连接
        x = self.layer12(x)
        return x
