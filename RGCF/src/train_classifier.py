import torch
import gym
import Server
import Agents
import TrainDiscrete_MNIST as TrainDiscrete
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
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import stats
import torch.optim as optim

from Classifier import Classifier


env = TrainDiscrete.CentralAgentEnvironment()
numWorkers = env.num_workers
episodes = 1
rewards = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Classifier(dataset = 'MNIST')
model = model.to(device)
# model.to(torch.double)


criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([10], device = device, dtype = torch.float32))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#training the classifier
for episode in range(1, episodes+1):
    done = False
    
    obs, id = env.reset()
    if(id < numWorkers):
        correct_action = 1
    else:
        correct_action = 0
    
    # score = 0 

    optimizer.zero_grad()
    
    obs = obs.astype('float32')
    obs = torch.from_numpy(obs)
    obs = obs.to(device)
    logit = model(obs)

    if(correct_action == 1):
        gt = 0
    else:
        gt = 1
    
    gt = torch.from_numpy(np.array([correct_action], dtype = np.float32))
    gt = gt.to(device)
    loss = criterion(logit, gt)
    loss.backward()
    optimizer.step()
    print('Classifier Loss', loss.item())

    while not done:
        obs, reward, done, info, id = env.step(correct_action)

        if(id < numWorkers):
            correct_action = 1
        else:
            correct_action = 0

        optimizer.zero_grad()

        obs = obs.astype('float32')
        obs = torch.from_numpy(obs)
        obs = obs.to(device)
        logit = model(obs)
        
        if(correct_action == 1):
            gt = 0
        else:
            gt = 1
        gt = torch.from_numpy(np.array([gt], dtype = np.float32))
        gt = gt.to(device)
        loss = criterion(logit, gt)
        loss.backward()
        optimizer.step()
        print('Classifier Loss', loss.item())
PATH = './classifiers/model_mnist_no_prior_with_inv.pth'
torch.save(model.state_dict(), PATH) 
