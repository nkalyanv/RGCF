import torch
from torch.nn import CrossEntropyLoss
from torch.distributions.uniform import Uniform
from Attacks import Attacks
import random

class Agent:
    def __init__(self, id, CentralServer, byz = False, attackType = None):
        self.centralServer = CentralServer
        self.parameterModel = None
        self.device = self.centralServer.device
        self.byz = byz
        self.id = id
        self.attackType = attackType
        
    
    def computeGradient(self, valid = False):
        self.parameterModel = self.centralServer.syncParameters()
        # for p in self.parameterModel.parameters():
        #     print(p)
        dataLoader, valid_loader = self.centralServer.sampleDataset()
        loss = 0
        criterion = CrossEntropyLoss()
        for (images, labels) in dataLoader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.parameterModel(images)
            loss += criterion(outputs, labels)
        loss.backward()
        grads = []
        #print(self.parameterModel.parameters())
        for p in self.parameterModel.parameters():
            grad = torch.clone(p.grad)
            if(self.byz == False): 
                grads.append(grad)
            else:
                #choice = random.choice(range(4))
                if(self.attackType == 1):
                    grads.append(Attacks.randomGaussianAttack(grad.shape).to(self.device))
                elif(self.attackType == 2):
                    grads.append(Attacks.inverseAttack(grad).to(self.device))
                elif(self.attackType == 3):  
                    grads.append(Attacks.allOnes(grad.shape).to(self.device))
                else:
                    grads.append(Attacks.gradShiftAttack(grad).to(self.device))

                #grads.append(torch.randn(p.grad.shape).to(self.device))
        valid_score = None
        if(valid == True):
            self.parameterModel.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for (images, labels) in valid_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.parameterModel(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            self.parameterModel.train()
            valid_score = (correct * 100 / total)
        if(valid):
            return loss, grads, valid_score
        return loss, grads

