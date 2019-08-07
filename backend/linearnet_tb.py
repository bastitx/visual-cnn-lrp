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
        self.outputLayers = [0, 3, 5, 7, 8]
    
    def forward(self, X, lrp_aware=False):
        return self.layers.forward(np.array(X))

    def relprop(self, R, lrp_var=None, param=None):
        return self.layers.lrp(np.array(R), lrp_var, param)
    
    def getActivations(self, layers=None):
        x = []
        if layers == None:
            layers = self.outputLayers
        for l in layers:
            if l == 0:
                x.append(self.layers.modules[0].X)
            else:
                x.append(self.layers.modules[l-1].Y)
        return x

    def getRelevances(self, layers=None):
        x = []
        if layers == None:
            layers = self.outputLayers
        for l in layers:
                x.append(self.layers.relevances[l])
        return x

    def getMetaData(self, X, layers=None):
        metadata = []
        self.forward(X)
        activations = self.getActivations(layers)
        for A in activations:
            metadata.append({ 'outputsize': A.shape })
        return metadata