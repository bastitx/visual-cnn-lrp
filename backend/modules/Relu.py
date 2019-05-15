import torch.nn as nn

class ReLU(nn.ReLU):
    def __init__(self):
        super().__init__()
    def relprop(self, R):
        return R
    def abrelprop(self, R, alpha):
        return R
