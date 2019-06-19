import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def gradprop(self, DY):
        DX = self.X * 0
        temp, indices = F.max_pool2d(self.X, self.kernel_size, self.stride,
                                     self.padding, self.dilation, self.ceil_mode, True)
        DX = F.max_unpool2d(DY, indices, self.kernel_size, self.stride, self.padding)
        return DX

    def relprop(self, R, epsilon=1e-9):
        Z = self.Y + epsilon*((self.Y >= 0).to(torch.float)*2 - 1)
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R

    def abrelprop(self, R, alpha):
        return self.relprop(R)
