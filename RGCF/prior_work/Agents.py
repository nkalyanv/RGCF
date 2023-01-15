import torch
from torch.nn import CrossEntropyLoss
from torch.distributions.uniform import Uniform
from Attacks import Attacks
import random

class Agent:
    def __init__(self, id, CentralServer, byz = False):
        self.centralServer = CentralServer
        self.parameterModel = None
        self.device = self.centralServer.device
        self.byz = byz
        self.id = id
        
    
    def computeGradient(self, valid = False, attackType = 0):
        self.parameterModel = self.centralServer.syncParameters()
        dataLoader = self.centralServer.sampleDataset()
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
                # print(p.grad)
                # break
            else:
                #choice = random.randint(2, 2)
                choice = attackType
                if(choice == 1):
                    grads.append(Attacks.randomGaussianAttack(grad.shape).to(self.device))
                elif(choice == 2):
                    grads.append(Attacks.inverseAttack(grad).to(self.device))
                elif(choice == 3):  
                    grads.append(Attacks.allOnes(grad.shape).to(self.device))
                else:
                    grads.append(Attacks.gradShiftAttack(grad).to(self.device))
        return loss, grads

