from Filter import Filter
import torch
from Krum import Krum
import statistics
import numpy as np

class Bulyan2(Filter):
    def __init__(self, f):
        Filter.__init__(self, f)

    def computeClosest(self, l, theta):
        l = l.sorted()
        mid = len(l) // 2
        median = (l[mid] + l[~mid]) / 2
        dists = [(abs(l[i] - median), i) for i in range(len(l))]
        dists = sorted(dists)
        ans = 0
        beta = int(theta - 2*self.n*self.f)
        for i in range(beta):
            ans += l[dists[i][1]]
        return ans / beta

    def abs(self, x):
        return x if x > 0 else -x

    # n is number of workers-first dimension
    # m is number of layers which is a list of tensors of funny shapes    
    def filter_grads(self, grad_list):
        f = self.f
        self.n = len(grad_list)
        self.m = len(grad_list[0])
        self.bad = int(self.n * f)
        theta = self.n - 2*self.bad
        S = []
        grad_list2 = grad_list.copy()
        while(len(S) < theta):
            krum = Krum(f)
            idx = krum.filter_grads(original_grad_list=grad_list2, flat = True, returnIndex=True)
            S.append(grad_list2[idx])
            del grad_list2[idx]
        shape = self.get_shape(S[0])
        S = list(map(self.expand_grads, S))

        grad_list_tensor = np.zeros((len(S), S[0].shape[0]))
        for i in range(len(S)):
            grad_list_tensor[i] = S[i].cpu().detach().numpy()

        grad_list_tensor = np.sort(grad_list_tensor, axis = 0)
        # print(grad_list_tensor.shape)

        mid = len(S) // 2

        for j in range(S[0].shape[0]):
            res = (grad_list_tensor[mid][j] + grad_list_tensor[~mid][j]) / 2
            ans = 0
            ptrLeft = mid
            ptrRight = mid + 1
            cnt = 0
            beta = theta - 2*self.n*self.f
            right = True
            while(cnt < beta):
                if(ptrRight >= len(S)):
                    right = False
                elif(ptrLeft <= -1):
                    right = True
                else:
                    right = False if (np.abs(grad_list_tensor[ptrLeft][j] - res) <= np.abs(grad_list_tensor[ptrRight][j] - res)) else True
                if(right):
                    ans += grad_list_tensor[ptrRight][j]
                    ptrRight += 1
                else:
                    ans += grad_list_tensor[ptrLeft][j]
                    ptrLeft -= 1
                cnt += 1

            S[0][j] = ans / beta

            # dist = [(abs(res - grad_list_tensor[i][j]), i) for i in range(len(S))]
            # dist = sorted(dist)
            # beta = int(theta - 2*self.n*self.f)
            # sum = 0
            # for i in range(beta):
            #     sum += grad_list_tensor[dist[i][1]][j]
            
            # # tmp = []
            # for i in range(len(S)):
            #     tmp.append(S[i][j])
            # ans = self.computeClosest(tmp, theta)
            # S[0][j] = ans

        return self.split_grads(torch.tensor(S[0], device=self.device), shape)
