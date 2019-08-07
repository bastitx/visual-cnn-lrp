from modules_tb import Module, Sequential, Convolution, Rect, MaxPool, Linear, Flatten, SoftMax
import numpy as np

class ConvNet(Module):
    def __init__(self, layers=None):
        super(ConvNet, self).__init__()
        if layers == None:
            self.layers = Sequential([
                Convolution((5,5,1,6),(1,1)),
                Rect(),
                MaxPool(),
                Convolution((5,5,6,16), (1,1)),
                Rect(),
                MaxPool(),
                Flatten(),
                Linear(4*4*16, 120),
                Rect(),
                Linear(120,100),
                Rect(),
                Linear(100,10),
                SoftMax()
            ])
        else:
            self.layers = layers
        self.outputLayers = [0, 2, 3, 5, 6, 8, 9, 11]
    
    def forward(self, X, lrp_aware=False):
        return self.layers.forward(np.pad(X.permute(0,2,3,1), ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=-1))

    def relprop(self, R, lrp_var=None, param=None):
        return self.layers.lrp(np.array(R), lrp_var, param)
    
    def getActivations(self, layers=None):
        x = []
        if layers == None:
            layers = self.outputLayers
        for l in layers:
            if l == 0:
                out = self.layers.modules[0].X
            else:
                out = self.layers.modules[l-1].Y
            if len(out.shape) == 4:
                x.append(np.transpose(out, (0, 3, 1, 2)))
            else:
                x.append(out)
        return x

    def getRelevances(self, layers=None):
        x = []
        if layers == None:
            layers = self.outputLayers
        for l in layers:
            out = self.layers.relevances[l]
            if len(out.shape) == 4:
                x.append(np.transpose(out, (0, 3, 1, 2)))
            else:
                x.append(out)
        return x

    def getMetaData(self, X, layers=None):
        metadata = []
        self.forward(X)
        activations = self.getActivations(layers)
        for A in activations:
            metadata.append({'outputsize': A.shape })
        return metadata