import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        n_maps = 64
        self.conv1 = nn.Conv2d(3, n_maps, 7, padding=1)
        self.conv2 = nn.Conv2d(n_maps, 2*n_maps, 7, padding=1)
        self.conv3 = nn.Conv2d(2*n_maps, 4*n_maps, 7, padding=1)
        self.conv4 = nn.Conv2d(4*n_maps, 8*n_maps, 7, padding=1)

    def forward(x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
        