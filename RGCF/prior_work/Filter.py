import torch
import numpy as np

class Filter():
    def __init__(self, f = 0.2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.f = f
 
    def filter_grads(self, grad_list, f):
        pass

    def get_shape(self, grad):
        m = len(grad)

        grad_shape = []
        sums = 0
        for i in range(m):
            shape = np.array(grad[i].shape)
            sums += np.prod(shape)
            grad_shape.append((sums, tuple(shape)))

        return grad_shape


    def expand_grads(self, grad_list):
        m = len(grad_list)
        
        cat = torch.zeros((1,), dtype = torch.float32, device = self.device)
        
        for j in range(m):
            cat = torch.cat((cat, grad_list[j].flatten()))

        return cat[1:]

    def split_grads(self, grad_list, grad_shape):
        split_list = []
        num_layers = len(grad_shape)
        
        split_list.append(torch.reshape(grad_list[:grad_shape[0][0]], grad_shape[0][1]))
        for i in range(1, num_layers):
            start = grad_shape[i-1][0]
            end = grad_shape[i][0]
            req_grad = grad_list[start:end]
            split_list.append(torch.reshape(req_grad, grad_shape[i][1]))

        return split_list

