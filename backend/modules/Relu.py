import torch.nn as nn

class ReLU(nn.ReLU):
    def __init__(self):
        super().__init__()
    def relprop(self, R, epsilon=1e-9):
        return R
    def abrelprop(self, R, alpha):
        return R
