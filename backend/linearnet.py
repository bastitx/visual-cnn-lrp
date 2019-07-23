from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules as lrp_module

class LinearNet(lrp_module.Module):
    def __init__(self):
        super(LinearNet, self).__init__([
            lrp_module.Reshape(28, 28, 1),
            lrp_module.Linear(28*28, 1296),
            lrp_module.ReLU(),
            lrp_module.Linear(1296, 1296),
            lrp_module.ReLU(),
            lrp_module.Linear(1296, 1296),
            lrp_module.ReLU(),
            lrp_module.Linear(1296, 10)
        ])