import torch
import torch.nn as nn

import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, dataset = 'MNIST'):
        super(Classifier, self).__init__()
        # an affine operation: y = Wx + b
        if (dataset == 'MNIST'):
            self.fc1 = nn.Linear(1199882, 64)  # 5*5 from image dimension
        elif (dataset == 'CIFAR'):
            self.fc1 = nn.Linear(1626442, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x