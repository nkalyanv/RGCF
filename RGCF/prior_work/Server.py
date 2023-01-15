import torch
from torch.optim import SGD
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from random import randint
from torch.nn import CrossEntropyLoss
import random 

class CentralServer:
    def __init__(self, dataset, batch_size, Net):
        self.agentList = {}
        self.Net = Net
        self.dataSet = dataset
        train_len = int(9/10 * len(self.dataSet))
        self.dataSet, self.dataValid = random_split(self.dataSet, (train_len, len(dataset) - train_len))
        self.valid_loader = torch.utils.data.DataLoader(self.dataValid, batch_size = batch_size, shuffle=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.parameterModel = Net()
        self.parameterModel = self.parameterModel.to(self.device)
        self.losses = []
        self.vals = []
        self.batch_size = batch_size
        self.learning_rate = 0.01
        self.optimizer = torch.optim.SGD(self.parameterModel.parameters(), lr=self.learning_rate)
        # self.id = 0
        # self.agents_losses = []
        self.criterion = CrossEntropyLoss()
        # self.valid = valid
            
    def addAgent(self, agent):
        self.agentList[agent.id] = agent 

    # def ResetPriors(self):
    #     # for layer in self.parameterModel.children():
    #     #     if hasattr(layer, 'reset_parameters'):
    #     #         layer.reset_parameters()
            
    #     self.id = 0
    #     self.losses = []
    
    def syncParameters(self):
        return self.parameterModel
    
    def sampleDataset(self):
        data_Train, _ = random_split(self.dataSet, (self.batch_size, len(self.dataSet) - self.batch_size))
        # print(data_Train[0][1], data_Train[1][1])
        train_loader = torch.utils.data.DataLoader(data_Train, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def compute_loss(self, valid = False):
        dataLoader = self.sampleDataset()
        loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for (images, labels) in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.parameterModel(images)
                loss += self.criterion(outputs, labels)
            if(valid == True):
                self.parameterModel.eval()
                for (images, labels) in self.valid_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.parameterModel(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                self.parameterModel.train()
        
        if(valid):
            self.losses.append(loss)
            self.vals.append((correct * 100 / total))
            return loss, (correct * 100 / total)
        
        self.losses.append(loss)
        return loss


    def gradStep(self, fgrad): 
        i = 0
        for p in self.parameterModel.parameters():
            p.grad = fgrad[i].clone()
            i+=1
        
        self.optimizer.step()


    def generateObservation(self, attack = 0):
        # self.id = randint(0, len(self.agentList) - 1)
        # agent = self.agentList[self.id]
        #ComputeGradient
        agents_grads = []
        # agents_losses = []
        for i in self.agentList:
            self.optimizer.zero_grad()  
            loss, grad = self.agentList[i].computeGradient(attackType = attack)
            
            # for i in range(len(grad)):
            #     grad[i] *= -1
            agents_grads.append(grad)
        random.shuffle(agents_grads)
        # agents_grads.reverse()
        # self.agents_losses.append(agents_losses)

        # print('Agent Grads:', agents_grads)
        return agents_grads


