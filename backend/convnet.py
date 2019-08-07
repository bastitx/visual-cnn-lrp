from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules as lrp_module

class ConvNet(lrp_module.Module):
    def __init__(self):
        super(ConvNet, self).__init__([
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
        ])
        self.outputLayers = [0, 2, 3, 5, 6, 9, 11, 12]