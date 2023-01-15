import torch

class Attacks:

    @staticmethod
    def randomGaussianAttack(dim, scale=1):
        return torch.randn(dim) * scale

    @staticmethod
    def inverseAttack(grad, scale = 1):
        return grad * -scale

    @staticmethod
    def gradShiftAttack(grad, scale = 50):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return grad + (torch.randn(grad.shape, device = device) * scale)

    @staticmethod
    def allOnes(dim, scale = 1e7):
        return torch.ones(dim) * scale