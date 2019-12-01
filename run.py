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

from cyclegan import CycleGAN

epochs = 20
model = CycleGAN()
data = "**********"

for epoch in epochs:
    for realA, realB in data:
        print(realA.shape)
        model.optimize_parameters(realA, realB)
        