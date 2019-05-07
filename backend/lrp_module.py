import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def relprop(self, R):
        V = torch.clamp(self.weight, min=0)
        Z = torch.mm(self.X, torch.transpose(V,0,1)) + 1e-9
        S = R / Z
        C = torch.mm(S, V)
        R = self.X * C
        return R

class ReLU(nn.ReLU):
    def __init__(self):
        super().__init__()
    def relprop(self, R):
        return R

class Reshape(nn.Module):
    def __init__(self, x, y, z):
        super(Reshape, self).__init__()
        self.x = x
        self.y = y
        self.z = z

    def forward(self, x):
        return x.view(-1, self.x*self.y*self.z)

    def relprop(self, R):
        return R.view(-1, self.z, self.y, self.x)

class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def gradprop(self, DY):
        DX = self.X * 0
        temp, indices = F.max_pool2d(self.X, self.kernel_size, self.stride,
                                     self.padding, self.dilation, self.ceil_mode, True)
        DX = F.max_unpool2d(DY, indices, self.kernel_size, self.stride, self.padding)
        return DX

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def gradprop(self, DY):
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] \
                                             - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv_transpose2d(DY, self.weight, stride=self.stride,
                                  padding=self.padding, output_padding=output_padding)

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R

class Softmax(nn.Softmax):
    def __init__(self):
        super().__init__()
    def relprop(self, R):
        return R
