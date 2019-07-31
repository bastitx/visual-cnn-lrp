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
    
    def forward(self, X, lrp_aware=False):
        return self.layers.forward(np.array(X.permute(0,2,3,1)))

    def relprop(self, R, lrp_var=None, param=None):
        return self.layers.lrp(np.array(R), lrp_var, param)
    
    def getActivations(self, layers=None):
        x = []
        if layers == None:
            layers = range(len(self.layers.modules)+1)
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
            layers = range(len(self.layers))
        for l in layers:
            out = self.layers.relevances[l]
            if len(out.shape) == 4:
                x.append(np.transpose(out, (0, 3, 1, 2)))
            else:
                x.append(out)
        return x

    def getMetaData(self, X):
        metadata = []
        metadata.append({ 'type': 'input2d', 'outputsize': [X.shape[0], X.shape[3], X.shape[1], X.shape[2]] })
        for layer in self.layers.modules:
            obj = {}
            X = layer.forward(X)
            obj['type'] = layer.__class__.__name__
            if len(X.shape) == 4:
                obj['outputsize'] = [X.shape[0], X.shape[3], X.shape[1], X.shape[2]]
            else:
                obj['outputsize'] = X.shape
            if obj['type'] == "Convolution":
                obj['kernelsize'] = layer.fh
            metadata.append(obj)
        return metadata