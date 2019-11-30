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
# what is instance normalization?
# what is patch gan?

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 9, 1)
        self.bn1 = nn.BatchNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm2d(128, affine=True)

        self.res11 = nn.Conv2d(128, 128, 3, padding=1) 
        self.resbn1 = nn.BatchNorm2d(128, affine=True)       
        self.res12 = nn.Conv2d(128, 128, 3, padding=1)
        self.res21 = nn.Conv2d(128, 128, 3, padding=1)
        self.resbn2 = nn.BatchNorm2d(128, affine=True)       
        self.res22 = nn.Conv2d(128, 128, 3, padding=1)
        self.res31 = nn.Conv2d(128, 128, 3, padding=1)
        self.resbn3 = nn.BatchNorm2d(128, affine=True)       
        self.res32 = nn.Conv2d(128, 128, 3, padding=1)
        self.res41 = nn.Conv2d(128, 128, 3, padding=1)
        self.resbn4 = nn.BatchNorm2d(128, affine=True)       
        self.res42 = nn.Conv2d(128, 128, 3, padding=1)
        self.res51 = nn.Conv2d(128, 128, 3, padding=1)
        self.resbn5 = nn.BatchNorm2d(128, affine=True)       
        self.res52 = nn.Conv2d(128, 128, 3, padding=1)
        self.res61 = nn.Conv2d(128, 128, 3, padding=1)   
        self.resbn6 = nn.BatchNorm2d(128, affine=True)       
        self.res62 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.bn4 = nn.BatchNorm2d(64, affine=True)
        self.conv5 = nn.ConvTranspose2d(64, 32, 3, 2    )
        self.bn5 = nn.BatchNorm2d(32, affine=True)
        self.conv6 = nn.ConvTranspose2d(32, 3, 9, 1)
        self.bn6 = nn.BatchNorm2d(3, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x1 = self.res11(x)
        x1 = self.resbn1(x1)
        x = self.res12(x + x1)

        x1 = self.res21(x)
        x1 = self.resbn2(x1)
        x = self.res22(x + x1)
        
        x1 = self.res31(x)
        x1 = self.resbn3(x1)
        x = self.res32(x + x1)
        
        x1 = self.res41(x)
        x1 = self.resbn4(x1)
        x = self.res42(x + x1)
        
        x1 = self.res51(x)
        x1 = self.resbn5(x1)
        x = self.res52(x + x1)
        
        x1 = self.res61(x)
        x1 = self.resbn6(x1)
        x = self.res62(x + x1)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = 255*F.tanh(x)
        return x
