import Server
import Agents
import gc
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms


class CentralAgentEnvironment():
    def __init__(self, filter, numWorkers = 1, numByzantine = 0, attack = 0):
        self.num_workers = numWorkers
        self.num_Byzantine = numByzantine
        self.attackType = attack
        self.reset_net()
        self.filter = filter
        

    def reset_net(self):
        class Net(nn.Module):    
            def __init__(self, num_gpus=0):
                super(Net, self).__init__()
                self.num_gpus = num_gpus

                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)

                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.max_pool2d(x, 2)

                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                # Move tensor to next device if necessary
                next_device = next(self.fc1.parameters())
                x = x.to(next_device)

                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = F.log_softmax(x, dim=1)
                return output
        model = Net()

        test = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
        centralServer = Server.CentralServer(test, 128,  Net)
        for i in range(self.num_workers):
            centralServer.addAgent(Agents.Agent(i, centralServer, False))

        for i in range(self.num_Byzantine):
            centralServer.addAgent(Agents.Agent(i + self.num_workers, centralServer, True))

        # self.stepCount = 1
        # self.numSteps = self.stepCount
        self.centralServer = centralServer
        self.parameterModel = self.centralServer.parameterModel        
        
    def step(self, valid = False):
        grads = self.centralServer.generateObservation(self.attackType)
        # print(grads)
        grads_mean = self.filter.filter_grads(grads)

        self.centralServer.gradStep(grads_mean)
        del grads, grads_mean
        if(valid):
            loss, acc = self.centralServer.compute_loss(valid)
            return loss, acc
        else:
            loss = self.centralServer.compute_loss(valid)  

        
        gc.collect()
        return loss

    def reset(self):
        self.reset_net()