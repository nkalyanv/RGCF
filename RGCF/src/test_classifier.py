import torch
import gym
import Server
import Agents
import TestDiscrete_MNIST as TestDiscrete
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import stats
import torch.optim as optim

from Classifier import Classifier


dataset = 'MNIST'
attack = 'none'
byz = str(90)
threshold = 0.35

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = '/home/imt2018034/AI/UpdatedCode/MARL/MARL-OpenAI/MARL/src/classifiers/model_mnist_no_prior.pth'
model = Classifier(dataset=dataset)
model = model.to(device)
model.load_state_dict(torch.load(PATH))

attackMap = {0 : 'shift', 1 : 'gauss', 2 : 'inv', 3 : 'ones'}
agentRatio = [(2, 8), (3,7), (5,5)]

for attackNum in range(4):
    for ratio in agentRatio:
        print(ratio, attackNum)
        attack = attackMap[attackNum]
        byz = int(((100 * ratio[0])/(ratio[0] + ratio[1])))
        env_test = TestDiscrete.CentralAgentEnvironment(numByzantine = ratio[0], numHonest = ratio[1], attackType = attackNum) 
        model.eval()
        sigmoid = torch.nn.Sigmoid()
        #testing the classifier
        episodes = 1
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        t_ = []

        numWorkers = env_test.num_workers
        for episode in range(1, episodes+1):
            done = False
            
            
            start = time.clock()
            obs, id = env_test.reset()
            if(id < numWorkers):
                correct_action = 1
            else:
                correct_action = 0
            
            score = 0 

            with torch.no_grad():
                obs = obs.astype('float32')
                obs = torch.from_numpy(obs)
                obs = obs.to(device)
                logit = model(obs)
            
            action_prob = sigmoid(logit)
            action_prob = action_prob.cpu().detach().numpy()

            if(action_prob[0] >= threshold):
                action = 0
            else:
                action = 1

            if(correct_action == action):
                if(action == 1):
                    tp += 1
                else:
                    tn += 1
            else:
                if(action == 1):
                    fp += 1
                else:
                    fn += 1

            t_.append(time.clock()-start)

            while not done:
                obs, reward, done, info, id = env_test.step(correct_action)
                score += reward
                if(id < numWorkers):
                    correct_action = 1
                else:
                    correct_action = 0

                with torch.no_grad():
                    obs = obs.astype('float32')
                    obs = torch.from_numpy(obs)
                    obs = obs.to(device)
                    logit = model(obs)
                
                gt = torch.from_numpy(np.array([correct_action], dtype = np.float32))
                gt = gt.to(device)

                action_prob = sigmoid(logit)
                action_prob = action_prob.cpu().detach().numpy()

                if(action_prob[0] >= threshold):
                    action = 0
                else:
                    action = 1
                
                if(correct_action == action):
                    if(action == 1):
                        tp += 1
                    else:
                        tn += 1
                else:
                    if(action == 1):
                        fn += 1
                    else:
                        fp += 1

                t_.append(time.clock()-start)
            # print('Episode:', episode, 'Reward:', score)
            # print('Vals:', env_test.centralServer.vals)
            # print('Losses:', env_test.centralServer.losses)

        print('True positive', tp)
        print('True Negative', tn)
        print('False Positive', fp)
        print('False Negative', fn)




        t_final = []
        for t in range(1, len(t_)):
            t_final.append(t_[t] - t_[t-1])
        t_final = np.array(t_final)
        print('Time', t_final)
        print('Avg', np.mean(t_final))
        print('std', np.std(t_final))
        exp_name = f'Correct_Results/{dataset}_{attack}_{byz}.txt'
        f = open(exp_name, "w")
        f.write('Classification Metrics\n')
        f.writelines(str([tp, tn, fp, fn]) + '\n')
        f.write('Val\n')
        f.writelines(str(env_test.centralServer.vals) + '\n')
        f.write('Loss\n')
        f.writelines(str(env_test.centralServer.losses) + '\n')
        f.close()