import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    resnet50,
    efficientnet_b4,
    convnext_base
)


class LeNet5(nn.Module):
    def __init__(self, pretrained=True):
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


def get_resnet18(pretrained=True):
    model = resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
    model.conv1.apply(weights_init)
    model.fc = nn.Linear(512, 2, bias=True)
    model.fc.apply(weights_init)
    return model


def get_resnet50(pretrained):
    model = resnet50(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
    model.conv1.apply(weights_init)
    model.fc = nn.Linear(2048, 2, bias=True)
    model.fc.apply(weights_init)
    return model


def get_enet(pretrained):
    model = efficientnet_b4(pretrained=pretrained)
    model.features[0][0] = nn.Conv2d(1,
                                     48,
                                     kernel_size=(3, 3),
                                     stride=(2, 2),
                                     padding=(1, 1),
                                     bias=False)
    model.features[0][0].apply(weights_init)
    model.classifier[1] = nn.Linear(1792, 2, bias=True)
    model.classifier[1].apply(weights_init)
    return model


def get_convnext(pretrained):
    model = convnext_base(pretrained=pretrained)
    model.features[0][0] = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
    model.features[0][0].apply(weights_init)
    model.classifier[2] = nn.Linear(1024, 2, bias=True)
    model.classifier[2].apply(weights_init)
    return model


def weights_init(m):
    init.kaiming_uniform_(m.weight.data)
