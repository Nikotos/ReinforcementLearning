import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from config import *


class QModel(nn.Module):

    def __init__(self):
        super(QModel, self).__init__()
        self.nl = nn.ReLU()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.lin1 = nn.Linear(56 * 56, 400)
        self.lin2 = nn.Linear(400, AMOUNT_OF_ACTIONS)

    def forward(self, x):
        x = self.nl(self.conv1(x))
        x = self.nl(self.conv2(x))
        x = self.nl(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.nl(self.lin1(x))
        x = self.lin2(x)
        return x
