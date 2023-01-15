from Filter import Filter 
import math
import torch

class SimpleFilter(Filter):
    def __init__(self, f = 0):
        Filter.__init__(self)
   
    def filter_grads(self, grad_list, f = 0):
        return grad_list[0]
        grad_shape = self.get_shape(grad_list[0])

        n = len(grad_list)

        d = 10000 * 2
        grad_list = list(map(self.expand_grads, grad_list))
        for i in range(len(grad_list)):
            grad_list[i] = grad_list[i][:d]
        

        f = float(2 / 3)

        S = grad_list[:math.ceil(f*n)]
        g_mean = torch.zeros_like(grad_list[0], device = self.device)

        for g in S:
            g_mean += g
        
        g_mean /= len(S)

        Cov = torch.zeros((d,d), device = self.device)

        for g in S:
            Cov += (torch.mm(torch.unsqueeze(g-g_mean, dim = 1), torch.unsqueeze(g-g_mean, dim = 1).T))
        Cov /= len(S)

        print(torch.max(Cov))

        grad_list = list(map(lambda g : self.split_grads(g, grad_shape), grad_list))

        return grad_list[0]
