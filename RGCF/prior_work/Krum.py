from Filter import Filter
import torch
import math

class Krum(Filter):
    def __init__(self, f):
        Filter.__init__(self, f)
        
        
    def distVec(grad1, grad2):
        return torch.sqrt(torch.sum((grad1 - grad2)**2))
    
    def filter_grads(self, original_grad_list, flat = True, returnIndex = False):
        out_grad = None
        best_Dist = float('inf')
        grad_list = original_grad_list.copy()
        n = len(grad_list)
        m = len(grad_list[0])
        
        # print(grad_list[1][0])
        # print(grad_list[0]0
        if(flat == True):
            grad_list = list(map(self.expand_grads, grad_list))
        
       
        idx = 0
        
        for i in range(n):
            dist = torch.zeros(n)
            # print(grad_list[i][:10])
            for j in range(n):
                if(i == j):
                    dist[i] = float('inf')
                    continue
                dist[j] = torch.dist(grad_list[i], grad_list[j], 2) 
            dist, _ = torch.sort(dist)
            val = torch.sum(dist[:n - math.ceil(self.f*n) - 2])
            
            if(val < best_Dist):
                best_Dist = val
                out_grad = original_grad_list[i]
                idx = i
        if(returnIndex):
            return idx
        return out_grad