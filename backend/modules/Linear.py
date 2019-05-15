import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = torch.mm(S, self.weight)
        R = self.X * C
        return R

    def gradprop(self, DY):
        return torch.mm(DY, self.weight)

    def abrelprop2(self, R, alpha):
        beta = 1 - alpha
        Z = self.weight.t().unsqueeze(0) * self.X.unsqueeze(-1)
        if not alpha == 0:
            Zp = torch.where(Z>0, Z, torch.zeros_like(Z))
            Zsp = torch.where(self.bias>0, self.bias, torch.zeros_like(self.bias)).unsqueeze(0).unsqueeze(0)
            Zsp += Zp.sum(1).unsqueeze(1)
            Ralpha = alpha * (Zp / Zsp * R.unsqueeze(1)).sum(2)
        else:
            Ralpha = 0
        if not beta == 0:
            Zn = torch.where(Z<0, Z, torch.zeros_like(Z))
            Zsn = torch.where(self.bias<0, self.bias, torch.zeros_like(self.bias)).unsqueeze(0).unsqueeze(0)
            Zsn += Zn.sum(1).unsqueeze(1)
            Rbeta = beta * (Zn / Zsn * R.unsqueeze(1)).sum(2)
        else:
            Rbeta = 0
        return Ralpha + Rbeta

    def abrelprop(self, R, alpha):
        beta = 1 - alpha
        X = self.X + 1e-9
        pself = Linear(self.in_features, self.out_features)
        dict = self.state_dict()
        dict['weight'] = torch.max(torch.zeros_like(dict['weight']), dict['weight'])
        dict['bias'] *= 0
        pself.load_state_dict(dict)
        nself = Linear(self.in_features, self.out_features)
        dict = self.state_dict()
        dict['weight'] = torch.min(torch.zeros_like(dict['weight']), dict['weight'])
        dict['bias'] *= 0
        nself.load_state_dict(dict)
        Zp = pself.forward(X)
        Sp = alpha * R / Zp
        Zn = nself.forward(X)
        Sn = beta * R / Zn
        R = X * (pself.gradprop(Sp) + nself.gradprop(Sn))
        return R
