from train_env_MNIST import CentralAgentEnvironment
import numpy as np
from Krum import Krum 
from SimpleFilter import SimpleFilter
#from Bulyan import Bulyan 
from Bulyan2 import Bulyan2
from Trimmed import TrimmedMean
from Trimmed import Median
import timeit
import math 


def run(env, name, percentage, attack):
    episodes = 1
    for episode in range(1, episodes+1):
        env.reset()
        num_steps = 500
        accL = []
        lossL = []
        losses = np.zeros(num_steps)
        start = timeit.timeit()
        times = []
        for i in range(num_steps):
            if((i+1) % 5 == 0):
                loss, acc = env.step(True)
                print(f'Episode: {episode}, Step: {i}, Loss: {loss}, Accuracy: {acc}')
                lossL.append(loss)
                accL.append(acc)
                end = timeit.timeit()
                times.append(end - start)
            else:
                loss = env.step()
                print(f'Episode: {episode}, Step: {i}, Loss: {loss}')
            losses[i] = loss
            if(loss > 100):
                break
            
        total_loss = np.sum(losses)
        print(total_loss)
        print(f'End of Episode: {episode}, Loss = {total_loss}')
        pathLoss = 'ExperimentResults/Loss_' + name + '_' + str(percentage) + '_' + attack
        pathAcc = 'ExperimentResults/Acc_' + name + '_' + str(percentage) + '_' + attack
        pathTime = 'ExperimentResults/Time_' + name + '_' + str(percentage) + '_' + attack
                
        np.save(pathLoss, lossL)
        np.save(pathAcc, accL)
        np.save(pathTime,times)
        print(pathLoss)



def_list = [(TrimmedMean, "TrimmedMean"), (Krum, "Krum"), (Median, "Median"), (Bulyan2, "Bulyan"), (SimpleFilter, "Optimal")]
ratio = [(2, 8), (3,7), (5,5)]
attacks = [(0, "gradShift"), (1, "randomGaussian"), (2, "inverseAttack"), (3, "allOnes")]
for defense in def_list:
    for (numB, numW) in ratio:
        for (a, attack) in attacks:
            env = CentralAgentEnvironment(defense[0](numB / (numB+numW)), numWorkers=numW, numByzantine=numB, attack = a)
            run(env, defense[1], int(((100 * numB)/(numW + numB))), attack)
            del env