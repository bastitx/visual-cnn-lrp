from modules_tb import Module, Sequential, Tanh, Linear, Flatten, SoftMax
import numpy as np

class LinearNet(Module):
    def __init__(self, layers=None):
        super(LinearNet, self).__init__()
        if layers == None:
            self.layers = Sequential([
                Flatten(),
                Linear(784, 1296),
                Tanh(),
                Linear(1296, 1296),
                Tanh(),
                Linear(1296, 1296),
                Tanh(),
                Linear(1296, 10),
                SoftMax()
            ])
        else:
            self.layers = layers
            self.layers.modules.insert(0, Flatten())
    
    def forward(self, X, lrp_aware=False):
        return self.layers.forward(X)

    def lrp(self, R, lrp_var=None, param=None):
        return self.layers.lrp(R, lrp_var, param)
    
    def getActivations(self, layers=None):
        x = []
        if layers == None:
            layers = range(len(self.layers.modules)+1)
        for l in layers:
            if l == 0:
                x.append(self.layers.modules[0].X)
            else:
                x.append(self.layers.modules[l-1].Y)
        return x

    def getRelevances(self, layers=None):
        x = []
        if layers == None:
            layers = range(len(self.layers))
        for l in layers:
                x.append(self.layers.relevances[l])
        return x

    def getMetaData(self, X):
        metadata = []
        metadata.append({ 'type': 'input2d', 'outputsize': X.shape })
        for layer in self.layers.modules:
            obj = {}
            X = layer.forward(X)
            obj['type'] = layer.__class__.__name__
            obj['outputsize'] = X.shape
            if obj['type'] == "Convolution":
                obj['kernelsize'] = layer.fh
            metadata.append(obj)
        return metadata