import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def abrelprop(self, R, alpha):
        beta = 1 - alpha
        X = self.X + 1e-9
        pself = Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        dict = self.state_dict()
        dict['weight'] = torch.max(torch.zeros_like(dict['weight']), dict['weight'])
        dict['bias'] *= 0
        pself.load_state_dict(dict)
        pself.X = X
        pself.Y = self.Y
        nself = Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        dict = self.state_dict()
        dict['weight'] = torch.min(torch.zeros_like(dict['weight']), dict['weight'])
        dict['bias'] *= 0
        nself.X = X
        nself.Y = self.Y
        nself.load_state_dict(dict)
        Zp = pself.forward(X)
        Sp = alpha * R / Zp
        Zn = nself.forward(X)
        Sn = beta * R / Zn
        R = X * (pself.gradprop(Sp) + nself.gradprop(Sn))
        return R
