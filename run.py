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

from gen import Generator
from disc import Discriminator


def train():
    GenAtoB = Generator()
    GenBtoA = Generator()
    DiscA = Discriminator()
    DiscB = Discriminator()


def forward(GenAtoB, GenBtoA):
    
    

if __name__ == '__main__':
    pass
