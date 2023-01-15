import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import matplotlib

attacks = [("gradShift"), ("randomGaussian"), ("inverseAttack"), ("allOnes")]
def_list = [("TrimmedMean"), ("Krum"), ("Median"), ("Bulyan")]
percentages = [20, 33, 50]

accuracies = {}
losses = {}

for attack in attacks:
    for percentage in percentages:
        for defense in def_list:
            pathLoss = 'ExperimentResults/Loss_' + defense + '_' + str(percentage) + '_' + attack + ".npy"
            pathAcc = 'ExperimentResults/Acc_' + defense + '_' + str(percentage) + '_' + attack + ".npy"
            pathTime = 'ExperimentResults/Time_' + defense + '_' + str(percentage) + '_' + attack + ".npy"
            if(defense == "Bulyan" and percentage != 20):
                continue
            losses[(attack, percentage, defense)] = (list(map(lambda x: x.item(), list(np.load(pathLoss, allow_pickle = True)))))
            accuracies[(attack, percentage, defense)] = (list(map(lambda x: x.item(), list(np.load(pathAcc, allow_pickle = True)))))


attacks = [("shift"), ("rand"), ("inv"), ("ones")]
percentages = [20, 33, 50]

for attack in attacks:
    for percentage in percentages:
        pathName = "../src/Correct_Results/MNIST_" + attack + "_" + str(percentage) + ".txt"
        f = open(pathName, 'r')
        s = f.read().split('\n')
        acc = np.array(list(map(float, s[3].strip('][').split(', '))))
        loss = np.array(list(map(float, s[5].strip('][').split(', '))))
        att = None
        if(attack == "shift"):
            att = "gradShift"
        elif(attack == "inv"):
            att = "inverseAttack"
        elif(attack == "gauss"):
            att = "randomGaussian"
        elif(attack == "ones"):
            att = "allOnes"
        idx = np.round(np.linspace(0, loss.shape[0] - 1, 100)).astype(int)
        losses[(att, percentage, 'RGCF')] = loss[idx]
        accuracies[(att, percentage, 'RGCF')] = acc



attacks = [("gradShift"), ("randomGaussian"), ("inverseAttack"), ("allOnes")]
percentages = [20, 33, 50]
def_list = [("RGCF"), ("TrimmedMean"), ("Krum"), ("Median"), ("Bulyan")]
colourMap = {'RGCF' : 'red', 'TrimmedMean' : 'blue', "Krum" : 'green', 'Median' : 'blue', 'Bulyan' : 'black'}
lineMap = {'RGCF' : 'solid', 'TrimmedMean' : 'solid', "Krum" : 'solid', 'Median' : 'solid', 'Bulyan' : 'solid'}

for attack in attacks:
    for percentage in percentages:
        for defense in def_list:
            if(defense == "Bulyan" and percentage != 20):
                continue
            if(losses[(attack, percentage, defense)] == [] or np.isnan(losses[(attack, percentage, defense)][-1]) == True):
                continue
            plt.plot(losses[(attack, percentage, defense)], label = defense, color = colourMap[defense], linestyle = lineMap[defense])
        plt.grid()
        font = {'family' : 'normal',
        'size'   : 12}
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        matplotlib.rc('font', **font)
        plt.legend()
        plt.savefig("Graphs/" + attack + "_" + str(percentage) + ".jpg")
        plt.clf()
        plt.cla()