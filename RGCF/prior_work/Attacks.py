import torch
import numpy as np
class Attacks:

    @staticmethod
    def randomGaussianAttack(dim, scale = 1):
        return torch.randn(dim) * scale

    @staticmethod
    def inverseAttack(grad, scale = 1):
        # scale = np.random.randint(1,2)
        
        return grad * -scale

    @staticmethod
    def gradShiftAttack(grad, scale = 50):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        scale = np.random.randint(30,51)
        return grad + (torch.randn(grad.shape, device = device) * scale)

    @staticmethod
    def allOnes(dim, scale = 1e7):
        return torch.ones(dim) * scale