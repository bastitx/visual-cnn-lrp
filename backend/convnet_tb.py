from modules_tb import Module, Sequential, Convolution, Rect, MaxPool, Linear, Flatten, SoftMax

class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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
    
    def forward(self, X, lrp_aware=False):
        return self.layers.forward(X)

    def lrp(self, R, lrp_var=None, param=None):
        return self.layers.lrp(R, lrp_var, param)
    
    def getActivations(self, layers=None):
        x = [self.layers.modules[0].X]
        if layers == None:
            layers = range(len(self.layers.modules))
        for l in layers:
            x.append(self.layers.modules[l].Y)
        return x

    def getRelevances(self, layers=None):
        x = [self.layers.relevances[0]]
        if layers == None:
            layers = range(len(self.layers))
        for l in layers:
            x.append(self.layers.relevances[l+1])
        return x

    def getMetaData(self):
        metadata = []
        metadata.append({ 'type': 'input2d', 'outputsize': [1,1,28,28] })
        metadata.append({ 'type': 'conv2d', 'outputsize': [1,6,24,24], 'kernelsize': 5 })
        metadata.append({ 'type': 'max_pool2d', 'outputsize': [1,6,12,12] })
        metadata.append({ 'type': 'conv2d', 'outputsize': [1,16,8,8], 'kernelsize': 5 })
        metadata.append({ 'type': 'max_pool2d', 'outputsize': [1,16,4,4] })
        metadata.append({ 'type': 'linear', 'outputsize': [1,120] })
        metadata.append({ 'type': 'linear', 'outputsize': [1,100] })
        metadata.append({ 'type': 'linear', 'outputsize': [1,10] })
        return metadata