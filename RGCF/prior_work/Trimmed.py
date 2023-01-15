from Filter import Filter 
import math
import torch

class TrimmedMean(Filter):
    def __init__(self, f):
        Filter.__init__(self, f)
   
    def filter_grads(self, grad_list):
        grad_shape = self.get_shape(grad_list[0])
        n = len(grad_list)

        grad_list = list(map(self.expand_grads, grad_list))
        dim = grad_list[0].shape[0]

        grad_list_tensor = torch.zeros((n, dim), device = self.device)
        for i in range(n):
            grad_list_tensor[i] = grad_list[i]

        # print(grad_list_tensor.shape)
        grad_list_tensor, _ = torch.sort(grad_list_tensor, dim = 0)
        # print(grad_list_tensor.shape)

        beta = int((self.f)*n)
        # print(grad_list_tensor[beta:n-beta,:].shape)
        grad_list_tensor = torch.sum(grad_list_tensor[beta:n-beta, :], dim = 0) / (n - 2 * beta)

        return self.split_grads(grad_list_tensor, grad_shape)



class Median(Filter):
    def __init__(self,f):
        Filter.__init__(self)
   
    def filter_grads(self, grad_list):
        grad_shape = self.get_shape(grad_list[0])
        n = len(grad_list)

        grad_list = list(map(self.expand_grads, grad_list))
        dim = grad_list[0].shape[0]

        grad_list_tensor = torch.zeros((n, dim), device = self.device)
        for i in range(n):
            grad_list_tensor[i] = grad_list[i]
        
        del grad_list
        # print(grad_list_tensor.shape)
        grad_list_tensor, _ = torch.sort(grad_list_tensor, dim = 0)
        # print(grad_list_tensor.shape)

        grad_list_tensor = (grad_list_tensor[(n//2), :] + grad_list_tensor[~(n//2), :])/2

        return self.split_grads(grad_list_tensor, grad_shape)
