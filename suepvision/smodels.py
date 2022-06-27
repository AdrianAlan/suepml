import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2).apply(weights_init)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5).apply(weights_init)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 120).apply(weights_init)
        self.fc2 = nn.Linear(120, 84).apply(weights_init)
        self.fc3 = nn.Linear(84, 2).apply(weights_init)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def weights_init(m):
    init.xavier_uniform_(m.weight.data)
