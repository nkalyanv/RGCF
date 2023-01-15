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
import Agents
import matplotlib.pyplot as plt
from random import randint


class CentralServer:
    def __init__(self, dataset, batch_size, Net, val = False):
        self.agentList = {}
        self.Net = Net
        self.dataSet = dataset
        self.dataValid = None
        self.valid = val 
        if(self.valid == True):
            train_len = int(9/10 * len(self.dataSet))
            self.dataSet, self.dataValid = random_split(self.dataSet, (train_len, len(dataset) - train_len))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.parameterModel = Net()
        self.parameterModel = self.parameterModel.to(self.device)
        self.losses = []
        self.vals = []
        self.batch_size = batch_size
        self.learning_rate = 0.01
        self.optimizer = torch.optim.SGD(self.parameterModel.parameters(), lr=self.learning_rate)
        self.grad = None
        self.id = 0
        self.prevAgent = 1
        self.prevLoss = 0
        if(self.valid):
            self.valid_loader = torch.utils.data.DataLoader(self.dataValid, batch_size = batch_size, shuffle=True)
        # for p in self.parameterModel.parameters():
        #     self.grad.append(torch.zeros(p.shape))
            

    def addAgent(self, agent):
        #agent object, prior, cumulative prev reward, number of times gradient was computed
        self.agentList[agent.id] = [agent, 0.01, 0, 0] 

    def ResetPriors(self):
        for layer in self.parameterModel.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            
        for id in self.agentList:
            self.agentList[id][1] = 0.01
            self.agentList[id][2] = 0
            self.agentList[id][3] = 0
        self.id = 0
        self.prevAgent = 1
        self.prevLoss = 0
        self.losses = []
    
    def syncParameters(self):
        return self.parameterModel
    
    # def sampleDataset(self):
    #     data_Train, _ = random_split(self.dataSet, (self.batch_size, len(self.dataSet) - self.batch_size))
    #     train_loader = torch.utils.data.DataLoader(data_Train, batch_size=self.batch_size, shuffle=True)
    #     return train_loader

    def sampleDataset(self):
        
        data_Train, _ = random_split(self.dataSet, (self.batch_size, len(self.dataSet) - self.batch_size))
        train_loader = torch.utils.data.DataLoader(data_Train, batch_size=self.batch_size, shuffle=True)
        valid_loader = None
        if(self.valid):
            valid_loader = self.valid_loader
            
        return train_loader, valid_loader


    def gradStep(self, weight):
        weight = weight[0]
        if weight < 0:
          weight = 0
        if weight > 1:
          weight = 1
        if self.grad is None: 
            return 
        i = 0
        for p in self.parameterModel.parameters():
            #print(p.grad, self.grad[i])
            p.grad = self.grad[i]
            p.grad *= weight
            i+=1
        self.agentList[self.id][1] = weight
        self.optimizer.step()


    def generateObservation(self):
        self.id = randint(0, len(self.agentList) - 1)
        # print('Id: ', self.prevAgent)
        agent = self.agentList[self.id][0]
        self.agentList[self.id][3] += 1
        self.optimizer.zero_grad()
        #ComputeGradient
        if(self.valid and len(self.losses) % 5 == 0):
            loss, self.grad, val = agent.computeGradient(valid = True)
            self.vals.append(val)
        else:
            loss, self.grad = agent.computeGradient()
        reward = self.prevLoss - loss.item()
        self.prevLoss = loss.item()
        l = []
        for p in self.grad:
            l.append(p.cpu().detach().numpy().flatten())
        arr = np.hstack(l)
        
        self.agentList[self.prevAgent][2] = self.agentList[self.prevAgent][2] + ((reward - self.agentList[self.prevAgent][2]) / self.agentList[self.id][3])
        self.prevAgent = self.id
        # print("Loss", loss.item())
        self.losses.append(loss.item())
        return {'gradient' : arr, 'model_loss' : loss.item(), 'prior' : self.agentList[self.id][1], 'prevRewards' : self.agentList[self.id][2]}, reward, self.id


