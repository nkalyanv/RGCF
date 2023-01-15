import gym.spaces as spaces
import numpy as np
import random
from gym import Env
# import MARL.src.Server as Server
# import MARL.src.Agents as Agents
import Server
import Agents
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.nn import CrossEntropyLoss

class CentralAgentEnvironment(Env):
    def __init__(self):
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
        #model = model.to('cuda:0')

        test = datasets.MNIST('../data', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
        # print(test.data[0])
        centralServer = Server.CentralServer(test, 128,  Net)
        self.num_workers = 1
        for i in range(self.num_workers):
            centralServer.addAgent(Agents.Agent(i, centralServer, False))

        self.num_Byzantine = 1
        for i in range(self.num_Byzantine):
            centralServer.addAgent(Agents.Agent(i + self.num_workers, centralServer, True))

        self.stepCount = 1
        self.numSteps = self.stepCount
        self.centralServer = centralServer
        self.parameterModel = self.centralServer.parameterModel
        dim = 0
        l = []
        for p in self.parameterModel.parameters():
            l.append(p.to('cpu').detach().numpy().flatten())
        arr = np.hstack(l)
        
        a = (np.resize(arr, (-1, 1)))
        b = np.resize(np.atleast_1d(0), (1, 1))
        # c = np.resize(np.atleast_1d(0), (1, 1))
        # d = np.resize(np.atleast_1d(0), (1, 1))
        observation = np.concatenate((a, b), axis = 0) 
        observation = np.reshape(observation, (1, -1))    
        observation = np.squeeze(observation)
        self.discreteSize = 1
        self.action_space = spaces.Discrete(self.discreteSize + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation.shape, dtype=np.float32
        )
        print(len(self.action_space.shape))
        print(len(self.observation_space.shape))
        
        
    def step(self, action):
            action = float(action)
            action /= self.discreteSize
            print('Action', action)
            print('NumSteps', self.numSteps)

            self.centralServer.gradStep([action])
            obs, reward, id = self.centralServer.generateObservation()
            self.numSteps -= 1
            done = False
            if(self.numSteps == 0):
              print("Episode Complete!")
              done = True

            a = (np.resize(obs['gradient'], (-1, 1)))
            b = np.resize(np.atleast_1d(obs['model_loss']), (1, 1))
            # c = np.resize(np.atleast_1d(obs['prior']), (1, 1))
            # d = np.resize(np.atleast_1d(obs['prevRewards']), (1, 1))
            
#             print(a.shape, b.shape, c.shape)
            observation = np.concatenate((a, b), axis = 0) 
            observation = np.reshape(observation, (1, -1))
            observation = np.squeeze(observation)
            print("Reward", reward)
            
            if(np.isnan(reward) == True or np.isnan(observation).any()):
                reward = -np.inf
                done = True
                observation = np.zeros(observation.shape)
            elif(obs['model_loss'] > 100):
                done = True
            return observation, reward * 10, done, {}, id
      #return observation, reward, done, info

    def reset(self):
            self.centralServer.ResetPriors()
            self.numSteps = 500
#             self.numSteps = 1
            obs, reward, id = self.centralServer.generateObservation()

            a = (np.resize(obs['gradient'], (-1, 1)))
            b = np.resize(np.atleast_1d(obs['model_loss']), (1, 1))
            # c = np.resize(np.atleast_1d(obs['prior']), (1, 1))
            # d = np.resize(np.atleast_1d(obs['prevRewards']), (1, 1))
#             print(a.shape, b.shape, c.shape)
            observation = np.concatenate((a, b), axis = 0) 
            observation = np.reshape(observation, (1, -1))
            observation = np.squeeze(observation)
            return observation, id
      #return observation  # reward, done, info can't be included


    def render(self, mode="human", close=False):
        pass

    def close(self):
            return -1
    
    