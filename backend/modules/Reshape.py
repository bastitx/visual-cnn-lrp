import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, x, y, z):
        super(Reshape, self).__init__()
        self.x = x
        self.y = y
        self.z = z

    def forward(self, x):
        return x.view(-1, self.x*self.y*self.z)

    def relprop(self, R, epsilon=1e-9):
        return R.view(-1, self.z, self.y, self.x)

    def abrelprop(self, R, alpha):
        return R.view(-1, self.z, self.y, self.x)
