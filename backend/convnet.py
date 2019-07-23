from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules as lrp_module

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(
            lrp_module.Conv2d(1, 6, 5),
            lrp_module.ReLU(),
            lrp_module.MaxPool2d(2, 2),
            lrp_module.Conv2d(6, 16, 5),
            lrp_module.ReLU(),
            lrp_module.MaxPool2d(2, 2),
            lrp_module.Reshape(4, 4, 16),
            lrp_module.Linear(4*4*16, 120),
            lrp_module.ReLU(),
            lrp_module.Linear(120, 100),
            lrp_module.ReLU(),
            lrp_module.Linear(100, 10)
        )
        for layer in self.layers:
            layer.register_forward_hook(forward_hook)
        self.relevances = []

    def forward(self, x):
        y = F.log_softmax(self.layers(x), dim=1)
        return y

    def relprop(self, R, kind='simple', param=None):
        self.relevances = [R]
        for l in range(len(self.layers), 0, -1):
            if kind == 'alphabeta':
                if param == None:
                    param=2
                self.relevances.append(self.layers[l-1].abrelprop(self.relevances[-1], param))
            else:
                if param == None:
                    param=1
                self.relevances.append(self.layers[l-1].relprop(self.relevances[-1], param))
        self.relevances.reverse()
        return self.relevances[0]

    def getActivations(self, layers=None):
        x = []
        if layers == None:
            layers = range(len(self.layers)+1)
        for l in layers:
            if l == 0:
                x.append(self.layers[0].X)
            else:
                x.append(self.layers[l-1].Y)
        return x

    def getRelevances(self, layers=None):
        x = []
        if layers == None:
            layers = range(len(self.layers))
        for l in layers:
            x.append(self.relevances[l])
        return x

    def getMetaData(self, X):
        metadata = []
        metadata.append({ 'type': 'input2d', 'outputsize': X.size() })
        for l in range(len(self.layers)):
            obj = {}
            X = self.layers[l].forward(X)
            obj['type'] = self.layers[l].__class__.__name__
            obj['outputsize'] = X.size()
            if obj['type'] == "Conv2d":
                obj['kernelsize'] = self.layers[l].kernel_size
            metadata.append(obj)
        return metadata

def forward_hook(self, input, output):
    self.X = input[0]
    self.Y = output
